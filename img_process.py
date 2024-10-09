#!/usr/bin/env python3
import cv2 as cv  # type: ignore

from skimage.filters import frangi  # type: ignore
import subprocess
import numpy as np  # type: ignore

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # type: ignore
from imblearn.metrics import sensitivity_specificity_support  # type: ignore

import os
from PIL import Image  # type: ignore
import streamlit as st  # type: ignore

####
IMAGE_PATH = "examples/01_g.JPG"
GROUND_TRUTH_PATH = "examples/01_g.tif"
OUTPUT_DIR = "output/"
####


def remove_small_elements(image: np.ndarray, min_size: int) -> np.ndarray:
    components, output, stats, _ = cv.connectedComponentsWithStats(
        image, connectivity=8
    )

    sizes = stats[1:, -1]
    width = stats[1:, -3]
    height = stats[1:, -2]
    components -= 1

    result = np.zeros((output.shape))

    for i in range(0, components):
        if sizes[i] >= min_size and (width[i] > 150 or height[i] > 150):
            result[output == i + 1] = 255

    return result


def normalize(image: np.ndarray, factor: float) -> np.ndarray:
    image = image * (factor / image.max())
    return image


def remove_border(color_img: np.ndarray, image: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(hsv, lower, upper)

    return cv.bitwise_and(image, mask)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv.LUT(image, table)


def apply_morphology(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)  # Noise removal
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)  # Filling holes
    return closing


def process_input(path: str) -> np.ndarray:
    path = "input/original/" + path
    color_image = cv.imread(path)
    if color_image is None:
        raise FileNotFoundError(f'"{path}" not found')

    gray_image = color_image[:, :, 1]  # 1 - green channel

    clahe = cv.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_image)  # 2 - equalized (contrast enhanced)

    adaptive_thresh = cv.adaptiveThreshold(
        equalized, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )  # 3 - adaptive threshold

    denoised = cv.fastNlMeansDenoising(adaptive_thresh, None, 15)  # 4 - denoised

    vessels = normalize(frangi(denoised), 255).astype(np.uint8)  # 5 - vessels

    vessels = gamma_correction(
        vessels, 6
    )  # helps with the contrast , more the gamma value more the contrast

    _, thresh = cv.threshold(vessels, 0, 255, cv.THRESH_BINARY)  # 6 - threshold

    small_removed = remove_small_elements(thresh, 200).astype(
        np.uint8
    )  # 7 - small removed

    kernel_for_dilation = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    kernel_for_erosion = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated_image = cv.dilate(small_removed, kernel_for_dilation, iterations=1)
    cleaned_image = cv.erode(dilated_image, kernel_for_erosion, iterations=1)

    end_result = remove_border(color_image, cleaned_image)  # 8 - border removed

    # cv.imwrite(OUTPUT_DIR + "0gray.jpg", gray_image)
    # cv.imwrite(OUTPUT_DIR + "1equalized.jpg", equalized)
    # cv.imwrite(OUTPUT_DIR + "2denoised.jpg", denoised)
    # cv.imwrite(OUTPUT_DIR + "3adap.jpg", adaptive_thresh)
    # cv.imwrite(OUTPUT_DIR + "4vessels.jpg", vessels)
    # cv.imwrite(OUTPUT_DIR + "5thresh.jpg", thresh)
    # cv.imwrite(OUTPUT_DIR + "6small_removed.jpg", small_removed)
    # cv.imwrite(OUTPUT_DIR + "7dilated.jpg", dilated_image)
    # cv.imwrite(OUTPUT_DIR + "8cleaned.jpg", cleaned_image)

    st.image(gray_image, caption="Gray Image", use_column_width=True)
    st.image(equalized, caption="Equalized Image", use_column_width=True)
    st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True)
    st.image(denoised, caption="Denoised Image", use_column_width=True)
    st.image(vessels, caption="Vessels", use_column_width=True)
    st.image(thresh, caption="Threshold", use_column_width=True)
    st.image(small_removed, caption="Small Removed", use_column_width=True)
    st.image(dilated_image, caption="Dilated Image", use_column_width=True)
    st.image(cleaned_image, caption="Cleaned Image", use_column_width=True)

    return end_result


def is_binary(image: np.ndarray) -> bool:
    return ((image == 0) | (image == 255)).all()


def compare(image: np.ndarray, truth: np.ndarray) -> dict:
    if not is_binary(image):
        raise Exception("Image is not binary (0 or 255)")

    if not is_binary(truth):
        raise Exception("Ground truth is not binary (0 or 255")

    image = image.flatten()
    truth = truth.flatten()

    report = classification_report(truth, image, output_dict=True)

    accuracy = round(accuracy_score(truth, image), 2)
    sensivity = round(report["255"]["recall"], 2)
    specifity = round(report["0"]["recall"], 2)

    weighted_result = sensitivity_specificity_support(truth, image, average="weighted")

    weight_sensivity = round(weighted_result[0], 2)
    weight_specifity = round(weighted_result[1], 2)
    matrix = confusion_matrix(truth, image)

    return {
        "accuracy": accuracy,
        "sensivity": sensivity,
        "specifity": specifity,
        "weight_sensivity": weight_sensivity,
        "weight_specifity": weight_specifity,
        "matrix": matrix.flatten(),
    }


def visualise(image: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted_vessels = image == 255
    true_vessels = truth == 255

    predicted_correct = predicted_vessels & true_vessels

    green_predicted = np.zeros((*truth.shape, 3), dtype=np.uint8)
    green_predicted[true_vessels, :] = 255

    green_predicted[predicted_correct] = [0, 255, 0]

    return green_predicted


st.title("Medical Image Processing with Gabor Filters and Morphology")
st.write("Upload an image to process with novel techniques and visualize the results.")

uploaded_image = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "tif"]
)

uploaded_truth = st.file_uploader(
    "Choose the ground truth image...", type=["jpg", "jpeg", "png", "tif"]
)

if (
    uploaded_image is not None and uploaded_truth is not None
):  # if both images are uploaded
    img = np.array(Image.open(uploaded_image))
    truth = np.array(Image.open(uploaded_truth).convert("L"))

    # Process the uploaded image
    processed_img = process_input(uploaded_image.name)

    # Compare with the ground truth
    comparison = compare(processed_img, truth)
    visualisation = visualise(processed_img, truth)

    # Display images side by side
    st.image(img, caption="Original", use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(processed_img, caption="Processed", use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(visualisation, caption="Visualization", use_column_width=True)

    # Display metrics
    st.write("Comparison Metrics")
    st.json(comparison)

    # Option to save the processed image
    save_option = st.checkbox("Save processed image")
    if save_option:
        out_path = (
            OUTPUT_DIR + "output_" + uploaded_image.name
        )  # save the processed image in the output directory
        cv.imwrite(out_path, processed_img)
        st.write(f"Processed image saved at {out_path}")

    subprocess.run(["python", "tortuosity.py"])

# if __name__ == "__main__":
#     img = process_input(IMAGE_PATH)
#     truth = cv.imread(GROUND_TRUTH_PATH, cv.IMREAD_GRAYSCALE)

#     comparison = compare(img, truth)
#     visualisation = visualise(img, truth)

#     if not os.path.isdir(OUTPUT_DIR):
#         os.mkdir(OUTPUT_DIR)
#         print(f'"{OUTPUT_DIR}" directory has been created.')

#     out_path = OUTPUT_DIR + "output.jpg"
#     cv.imwrite(out_path, img)
#     print(f'"{out_path}" has been saved.')

#     vis_path = OUTPUT_DIR + "visualisation.jpg"
#     cv.imwrite(vis_path, visualisation)
#     print(f'"{vis_path}" has been saved.')

#     print(comparison)
