"""
Real-Time Digit Recognition with Streamlit and OpenCV.

This application uses OpenCV and Streamlit to capture a live webcam feed, detect handwritten 
digits in real-time, preprocess them, and classify them using a trained CNN model. 

Features
--------
- Captures video frames from a webcam.
- Preprocesses images for digit recognition:
  - Converts to grayscale.
  - Binarizes using Otsu's thresholding.
  - Applies morphological operations to clean noise.
- Extracts digit contours and resizes them to 28x28 pixels.
- Uses a trained CNN model to predict the digit.
- Displays the bounding box and predicted digit on the video stream.

Usage
-----
Run the application with:
    >>> streamlit run app.py

Dependencies
------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Streamlit (`streamlit`)
- scikit-learn (`sklearn`)
- PIL (`Pillow`)
- A trained MNIST CNN classifier

Example
-------
>>> python -m streamlit run app.py
"""

import os
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from mnist_cnn_classifier import MnistCnnClassifier

import os
import streamlit as st

# Get the directory where the script is running
current_dir = os.path.dirname(__file__)

# List all subdirectories
st.write("📂 **Subdirectories in Current Directory:**")
st.write([d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))])

# List files in the subdirectory (replace 'models' with your actual folder name)
subdir = "models"  # Change this to match your actual folder
subdir_path = os.path.join(current_dir, subdir)

if os.path.exists(subdir_path):
    st.write(f"📂 **Files in `{subdir}` Folder:**")
    st.write(os.listdir(subdir_path))
else:
    st.error(f"🚨 Subdirectory `{subdir}` NOT FOUND!")


# Load the trained model
model = MnistCnnClassifier()
model.load_model()

# Constants
kernel = np.ones((4, 4), np.uint8)
contour_color = (0, 255, 255)
image_border = 40

# Define functions
def preprocess_image(img):
    """
    Preprocesses an input image for digit recognition.

    This function converts an image to grayscale, applies binary thresholding 
    using Otsu's method, and performs morphological operations to remove noise.

    Parameters
    ----------
    img : np.ndarray
        The input image in BGR format.

    Returns
    -------
    np.ndarray
        The preprocessed binary image.
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Remove noise
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    # Thicken strokes
    img_binary = cv2.dilate(img_binary, kernel, iterations=4)
    return img_binary

def resize_and_center(img):
    """
    Resizes and centers a binary digit image to 28x28 pixels.

    The function scales the digit to fit within a 20x20 bounding box while 
    maintaining the aspect ratio. It then centers the digit using image 
    moments and applies translation.

    Parameters
    ----------
    img : np.ndarray
        The preprocessed binary digit image.

    Returns
    -------
    np.ndarray
        A 28x28 image with the digit centered.
    """
    # Scale the digit to fit within a 20x20 bounding box
    height, width = img.shape
    scale = 20.0 / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    img = cv2.resize(img, (new_width, new_height))

    # Place the new 20x20 image inside a blank 28x28 image
    new_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset, y_offset = (28 - new_width) // 2, (28 - new_height) // 2
    new_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img

    # Center the image
    moments = cv2.moments(new_img)
    center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else 14
    center_y = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else 14
    shift_x, shift_y = 14 - center_x, 14 - center_y
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_img = cv2.warpAffine(new_img, translation_matrix, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return centered_img

# Initialize Streamlit UI
st.markdown("<h3 style='text-align: center;'>Sifferigenkänning i realtid</h3>", unsafe_allow_html=True)

st.sidebar.header("Kontroller")

# Use session state to track webcam status
if "capture_active" not in st.session_state:
    st.session_state.capture_active = False

# Buttons to start and stop webcam
if st.sidebar.button("Starta kameran"):
    st.session_state.capture_active = True

if st.sidebar.button("Stoppa Kameran"):
    st.session_state.capture_active = False

# Create a placeholder for the video feed
frame_holder = st.empty()

# Webcam loop
if st.session_state.capture_active:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while st.session_state.capture_active:
        success, img = video_capture.read()
        if not success:
            st.error("Failed to access webcam.")
            break
        
        # Preprocess image
        preprocessed_img = preprocess_image(img)
        # Find the contours in the image
        contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate over the contours
        for contour in contours:
            # Get a bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Ignore small or contours too close to the border
            if x < image_border or x + w > 1040 or y < image_border or y + h > 680 or h < 72:
                continue
            
            # Ectract the detected digit
            cropped_img = preprocessed_img[y: y + h, x: x + w]
            
            # Apply padding to maintain aspect ratio
            r = max(w, h)
            y_pad = ((w - h) // 2 if w > h else 0) + r // 5
            x_pad = ((h - w) // 2 if h > w else 0) + r // 5
            cropped_img = cv2.copyMakeBorder(cropped_img, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=0)
            
            # Resize and center the digit
            centered_img = resize_and_center(cropped_img)
            # reshape for prediction
            image_for_prediction = centered_img.reshape(1, -1)
            # Make prediction
            pred = model.predict(image_for_prediction)[0]
            
            # Draw bounding box and prediction text
            cv2.rectangle(img, (x, y), (x + w, y + h), contour_color, 2)
            cv2.putText(img, str(pred), (x + w // 5, y - h // 5), cv2.FONT_HERSHEY_SIMPLEX, 1, contour_color, 2)

        # Convert to RGB for Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_holder.image(img_rgb, channels="RGB", use_column_width=True)

    # Release the webcam when stopped
    video_capture.release()
    cv2.destroyAllWindows()