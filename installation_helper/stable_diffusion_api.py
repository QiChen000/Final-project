import requests
import json
from io import BytesIO
from PIL import Image
from requests.auth import HTTPBasicAuth
import base64
import cv2
import mediapipe as mp
import numpy as np


def image_to_base64_string(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def get_segmentation_mask(image):
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Load and resize the image
    original_image = image
    resized_image = cv2.resize(original_image, (1024, 1024))

    # Process the image for segmentation
    results = selfie_segmentation.process(
        cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    )

    # Create the mask image from segmentation results
    mask_image = (results.segmentation_mask > 0.5) * 255
    mask_image = mask_image.astype(np.uint8)

    cv2.imwrite("./mask.jpg", mask_image)
    cv2.imwrite("./original.jpg", resized_image)

    # Convert images to Base64 strings
    original_image_string = image_to_base64_string(resized_image)
    mask_image_string = image_to_base64_string(mask_image)

    # Cleanup
    selfie_segmentation.close()

    return original_image_string, mask_image_string


def send_request_to_sd(img, mask_img):
    url = "http://79.117.50.54:21123/" 
    full_url = url + "sdapi/v1/img2img" 
    # If you have specific headers or parameters, add them here
    headers = {
        "Content-Type": "application/json",
        # Add other headers if necessary
    }

    json_data = {
        "prompt": "lion,animal,nudity",
        "width": 1024,
        "height": 1024,
        "init_images": [img],
        "mask_images": mask_img,
        "steps": 30,
        "cfg_scale": 9,
        "negative_prompt": "",
        # ... (include all other key-value pairs from your JSON here)
        "alwayson_scripts": {},
    }
    # Basic Authentication
    username = "user"  # Replace with the actual username
    password = "password"  # Replace with the actual password

    # If it's a POST request, you might send data like this:
    # data = {'key': 'value'}

    try:
        response = requests.post(
            full_url,
            headers=headers,
            json=json_data,
            auth=HTTPBasicAuth(username, password),
        )

        if response.status_code == 200:
            base64_image_key = "images"  # Change this to the actual key
            base64_image_str = response.json().get(base64_image_key)[0]

            if base64_image_str:
                image_data = base64.b64decode(base64_image_str)
                with open("output_image.jpg", "wb") as f:
                    f.write(image_data)
                print("Image downloaded successfully.")
            else:
                print("No image data in response")

        else:
            print("Error in API request:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    # Load the image
    image = cv2.imread("../test.jpg")

    # Resize the image
    resized_image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)

    # Get the mask image
    resized_image_string, mask_image_string = get_segmentation_mask(resized_image)

    # Send the request to SD
    send_request_to_sd(resized_image_string, mask_image_string)
