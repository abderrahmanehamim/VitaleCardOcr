import streamlit as st
import cv2
import paddleocr
from paddleocr import PaddleOCR
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageOps
import re
import numpy as np
import time


# Load YOLO model
model = YOLO(r"VitaleCardapp\best2.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='fr', show_log=False)

# Preprocessing function for the card number region
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (2 * image.shape[1], 2 * image.shape[0]))
    thresh = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 5)
    return thresh

def process_image(image_path):
    # Load the cropped Vitale card image
    original_image = cv2.imread(image_path)
    
    results0 = model.predict(source=original_image, classes=[3])
    try:
        bbox2 = results0[0].boxes.xyxy[0]
        x5, y5, x6, y6 = map(int, bbox2)
        cropped_image = original_image[y5:y6, x5:x6]
    except IndexError:
        st.error("No detections found for the card vitale.")
        return

    # Use YOLOv8 to get the bounding box of the card number
    results = model.predict(source=cropped_image, classes=[0])
    # Use YOLOv8 to get the bounding box of the fullname
    results2= model.predict(source=cropped_image, classes=[2])

    try:
        bbox = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        cropped_number = cropped_image[y1:y2, x1:x2]
        preprocessed_number = preprocess_image(cropped_number)
        preprocessed_number_pil = Image.fromarray(cropped_number)
    except IndexError:
        st.error("No detections found for the card number.")
        return
    try:
        bbox1 = results2[0].boxes.xyxy[0]
        x3, y3, x4, y4 = map(int, bbox1)
        cropped_fullname = cropped_image[y3:y4, x3:x4]
        preprocessed_fullname = preprocess_image(cropped_fullname)
        preprocessed_fullname_pil = Image.fromarray(cropped_fullname)
    except IndexError:
        st.error("No detections found for the card Fullname.")
        return

    # Common rotation angles to try
    rotation_angles = [0, 90, 180, 270, 360]
    # Flag to track if the correct orientation is found
    correct_orientation_found = False

    # Try rotating the image with different angles
    for angle in rotation_angles:
        # Rotate the image
        rotated_number = preprocessed_number_pil.rotate(angle, resample=Image.BICUBIC, fillcolor=0, expand=True)
        # Convert the rotated_number to a numpy array
        rotated_number_array = np.array(rotated_number)
        # Use EasyOCR to extract text from the rotated image
        result1 = ocr.ocr(rotated_number_array, cls=True)
        extracted_text3 = []
        for box, txt in result1[0]:
            extracted_text3.append(txt[0])
        cleaned_result = ''.join(re.findall(r'\d', ''.join(extracted_text3)))
        # Check if conditions are satisfied
        if len(cleaned_result) >= 15 and cleaned_result[0] in ['1', '2']:
            # Rotate the full name image using the same angle
            rotated_fullname = preprocessed_fullname_pil.rotate(angle, resample=Image.BICUBIC, fillcolor=0, expand=True)
            # Use PaddleOCr to extract text from the rotated full name image
            fullname = ocr.ocr(np.array(rotated_fullname), cls=True)
            extracted_text4 = []
            for box, txt in fullname[0]:
                extracted_text4.append(txt[0])
            # Clean the first name and last name
            try:
                cleaned_firstname = re.sub(r'[^A-Za-zÀ-ÿ\'\-]', '', extracted_text4[0])
                cleaned_lastname = ''
                if len(extracted_text4) > 1:
                    cleaned_lastname = re.sub(r'[^A-Za-zÀ-ÿ\'\-]', '', extracted_text4[1])
                # Print the cleaned first name and last name
                st.success("Card Number: " + cleaned_result)
                st.success("First Name: " + cleaned_firstname)
                st.success("Last Name: " + cleaned_lastname)
            except IndexError:
                st.error("There's an error in cleaning the extracted text.")
            correct_orientation_found = True
            break

    if not correct_orientation_found:
        st.error("Could not find the correct orientation.")

# Streamlit UI
st.title("Vitale Card OCR")
image_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

if image_file is not None:
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_file.getvalue())

    image = Image.open(temp_image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    process_image(temp_image_path)

    # Delete the temporary image file
    os.remove(temp_image_path)
