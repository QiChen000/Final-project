import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
from installation_helper import (
    get_segmentation_mask,
    print_image_with_cups,
    prepare_image_for_printing,
    initialize_segmentation_model,
)
import webuiapi
from tkinter import *
import shutil
from datetime import datetime
import threading
from PIL import Image, ImageTk
from collections import deque

######### Please Check These Important Parameters Before Running #########
PRINTER_NAME = "HP_DeskJet_3630_series"  # The name of the printer you want to use
STILL_DURATION = 5  # seconds to the audience stillness check
THRESHOLD = 3  # pixels to the audience stillness check
HOST_IP = "31.12.82.146"  # The IP address of the Stable Diffusion server
HOST_PORT = 12589
USER_NAME = "qi"  # The username of the Stable Diffusion server
USER_PASSWORD = "10121012"  # The password of the Stable Diffusion server
DISTANCE_HISTORY_SIZE = 5
TRACKING_CONFIDENCE = 0.3

######################################################################
# Initialize a buffer to store the distance history
distance_history = deque(maxlen=DISTANCE_HISTORY_SIZE)

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")


###UI Funciton
def update_button_appearance(button, action, is_enabled):
    button.config(
        bg="green" if is_enabled else "red",
        text=f"{action} {'Enabled' if is_enabled else 'Disabled'}",
    )


# Toggle functions with label update
def toggle_printing(event=None):
    global print_enabled
    print_enabled = not print_enabled
    update_button_appearance(print_toggle_button, "Printing", print_enabled)


def toggle_tracking(event=None):
    global tracking_enabled
    tracking_enabled = not tracking_enabled
    update_button_appearance(tracking_toggle_button, "Tracking", tracking_enabled)


# Tkinter window setup
def setup_tkinter_window():
    global print_toggle_button, tracking_toggle_button, root

    root = Tk()
    root.title("Interactive Control Panel")
    root.protocol(
        "WM_DELETE_WINDOW",
        on_closing,
    )

    # Image display label
    image_label = Label(root)
    image_label.pack()

    # Custom toggle buttons using labels
    print_toggle_button = Label(root, bg="green", fg="white")
    print_toggle_button.pack(fill="x", padx=5, pady=5)
    print_toggle_button.bind("<Button-1>", toggle_printing)

    tracking_toggle_button = Label(root, bg="green", fg="white")
    tracking_toggle_button.pack(fill="x", padx=5, pady=5)
    tracking_toggle_button.bind("<Button-1>", toggle_tracking)

    # Initialize button text
    update_button_appearance(print_toggle_button, "Printing", print_enabled)
    update_button_appearance(tracking_toggle_button, "Tracking", tracking_enabled)

    # Start the video capture in a new thread
    threading.Thread(target=lambda: cam_capture_loop(image_label), daemon=True).start()
    print("Program started")
    root.mainloop()


def move_images_with_timestamp(source_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # Construct the full file path
            source_file = os.path.join(source_dir, filename)
            # Add timestamp to the filename
            dest_file = os.path.join(dest_dir, f"{timestamp}_{filename}")
            # Move the file
            shutil.move(source_file, dest_file)


def on_closing():
    move_images_with_timestamp("./tmp_img", "./storage_img")
    root.destroy()


def detect_humans(frame):
    results = model.track(
        frame,
        persist=True,
        conf=TRACKING_CONFIDENCE,
        classes=[0],
        imgsz=640,
        max_det=1,
        verbose=False,
    )
    if results[0].boxes:
        box = results[0].boxes.xyxy[0].numpy()
        return box
    return None


def is_human_still(current_box, previous_box, threshold=THRESHOLD):
    if previous_box is None:
        return False

    center_current = (
        (current_box[0] + current_box[2]) / 2,
        (current_box[1] + current_box[3]) / 2,
    )
    center_previous = (
        (previous_box[0] + previous_box[2]) / 2,
        (previous_box[1] + previous_box[3]) / 2,
    )
    distance = np.sqrt(
        (center_current[0] - center_previous[0]) ** 2
        + (center_current[1] - center_previous[1]) ** 2
    )

    # Add the distance to the deque
    distance_history.append(distance)

    # Calculate the median of distances
    median_distance = np.median(list(distance_history))
    print(
        f"Median distance over last {DISTANCE_HISTORY_SIZE} frames: {median_distance}"
    )

    return median_distance < threshold


def cam_capture_loop(label):
    # Initialize MediaPipe Selfie Segmentation
    print("Initializing MediaPipe Selfie Segmentation...")
    selfie_segmentation = initialize_segmentation_model()
    # Initalize WebSD APi
    api = webuiapi.WebUIApi(host=HOST_IP, port=HOST_PORT)
    api.set_auth(USER_NAME, USER_PASSWORD)
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera stream not available.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    previous_box = None
    stillness_start_time = None
    stillness_duration = STILL_DURATION  # Time in seconds to check for stillness
    image_counter = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read camera frame.")
                break

            h, w, _ = frame.shape
            start = int((w - h) / 2)
            cropped_frame = frame[:, start : start + h]

            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            if tracking_enabled:
                human_box = detect_humans(frame_rgb)
                if human_box is not None:
                    if is_human_still(human_box, previous_box):
                        if stillness_start_time is None:
                            stillness_start_time = time.time()
                        elif time.time() - stillness_start_time >= stillness_duration:
                            img, mask_img = get_segmentation_mask(
                                selfie_segmentation, cropped_frame, image_counter
                            )
                            inpainting_result = api.img2img(
                                images=[img],
                                mask_image=mask_img,
                                inpainting_fill=1,
                                steps=40,
                                seed=3661460071,
                                prompt="lion,fur,animal,nudity,female",
                                cfg_scale=8.0,
                                denoising_strength=0.7,
                            )
                            print_ready_image = prepare_image_for_printing(
                                inpainting_result.image, (10, 15), 150
                            )

                            print_options = {
                                "print-color-mode": "monochrome",
                                "fit-to-page": "True",
                                "media": "4x6",
                            }

                            print_image_with_cups(
                                print_ready_image,
                                PRINTER_NAME,
                                "Photo Print",
                                print_options,
                                image_counter,
                                print_enabled,
                            )
                            image_counter += 1
                            stillness_start_time = None
                    else:
                        stillness_start_time = None

                    previous_box = human_box
                    start_point = (int(human_box[0]), int(human_box[1]))
                    end_point = (int(human_box[2]), int(human_box[3]))
                    cv2.rectangle(cropped_frame, start_point, end_point, (0, 255, 0), 2)
            cropped_frame = cv2.resize(cropped_frame, (640, 480))
            cv_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv_rgb))
            # Update the Tkinter label
            label.config(image=photo)
            label.image = photo

    finally:
        cap.release()
        print("Closing camera stream...")


if __name__ == "__main__":
    # Initialize global toggle states
    print_enabled = FALSE
    tracking_enabled = FALSE
    # Start the Tkinter window
    setup_tkinter_window()
