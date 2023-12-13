import cv2
import cups
import numpy as np
import time


def resize_image_for_printing(image, width, height):
    # Convert the PIL image to a numpy array
    numpy_image = np.array(image)
    # Convert RGB to BGR
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    # Resize the image
    resized_image = cv2.resize(
        opencv_image, (width, height), interpolation=cv2.INTER_AREA
    )

    return resized_image


def prepare_image_for_printing(original_image, paper_size_inches, dpi):
    original_image = np.array(original_image)
    # Convert RGB to BGR
    opencv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    # Convert paper size to pixels
    paper_width_pixels = paper_size_inches[0] * dpi
    paper_height_pixels = paper_size_inches[1] * dpi

    # Resize the original image to fill the width of the paper
    resized_image = cv2.resize(original_image, (paper_width_pixels, paper_width_pixels))

    # Create a new blank image with white background
    blank_image = np.full(
        (paper_height_pixels, paper_width_pixels, 3), 255, dtype=np.uint8
    )

    # Calculate the position to place the resized image (to center it vertically)
    y_offset = (blank_image.shape[0] - resized_image.shape[0]) // 2

    # Place the resized image on the blank image
    blank_image[y_offset : y_offset + resized_image.shape[0], :] = resized_image

    return blank_image


def print_image_with_cups(
    image, printer_name, title, print_options=None, index=0, print_enabled=False
):
    # Save the resized image temporarily
    temp_image_path = f"./tmp_img/printing_{index}.jpg"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(temp_image_path, image)
    print(f"Saving the ready-to-print image to {temp_image_path}")
    if print_enabled:
        # Connect to CUPS and print the file
        conn = cups.Connection()
        job_id = conn.printFile(printer_name, temp_image_path, title, print_options)

        # Wait for the print job to complete
        while True:
            # Get the job info
            job_info = conn.getJobAttributes(job_id)

            # Check if the job is completed
            if job_info["job-state"] == 9:  # job-state 9 means 'completed'
                print("Print job completed.")
                break

            # Wait a bit before checking again
            time.sleep(1)
    else:
        print("Printing disabled.")


if __name__ == "__main__":
    # Connect to CUPS
    conn = cups.Connection()
    # Get the name of the default printer or specify a printer
    printers = conn.getPrinters()
    print(printers)
