#!/usr/bin/env python3
import cv2 as cv  # type: ignore

from skimage.filters import frangi  # type: ignore

from PIL import Image  # type: ignore

import numpy as np  # type: ignore

from skimage.filters import gabor  # type: ignore

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # type: ignore
from imblearn.metrics import sensitivity_specificity_support  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os
import seaborn as sns  # type: ignore

import streamlit as st  # type: ignore

####
IMAGE_PATH = "examples/01_g.JPG"
GROUND_TRUTH_PATH = "input/truth"
OUTPUT_DIR = "output/"
####


# To remove small elements from an image i.e noise removal
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


# Normalize the image (0-255)
def normalize(image: np.ndarray, factor: float) -> np.ndarray:
    image = image * (factor / image.max())
    return image


# Apply Gabor Filter to the image to enhance the features
def apply_gabor_filter_mutliscale(image: np.ndarray) -> np.ndarray:
    filtered_real, _ = gabor(image, frequency=0.6)
    return normalize(filtered_real, 255).astype(np.uint8)


# Apply Gabor Filter to the image to enhance the features
def apply_gabor_filter(image: np.ndarray) -> np.ndarray:
    gabor_kernel = cv.getGaborKernel((21, 21), 8.0, 0, 10.0, 0.5, 0, ktype=cv.CV_32F)
    filtered_image = cv.filter2D(image, cv.CV_8UC3, gabor_kernel)
    return filtered_image


# Apply Morphological Operations to the image to remove noise and fill holes
def apply_morphology(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)  # Noise removal
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)  # Filling holes
    return closing


# Remove border from the image using HSV color space and thresholding technique
def remove_border(color_img: np.ndarray, image: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(hsv, lower, upper)

    return cv.bitwise_and(image, mask)


# Process the input image to extract the vessels from it
def process_input(path: str) -> np.ndarray:
    # input/original/path
    path = f"input/original/{path}"
    color_image = cv.imread(path)
    if color_image is None:
        raise FileNotFoundError(f'"{path}" not found')

    gray_image = color_image[:, :, 1]

    # Contrast Limited Adaptive Histogram Equalization (CLAHE) for enhancing the contrast of the image and removing noise
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_image)

    # Apply Gabor filter to enhance the features of the image
    gabor_filtered = apply_gabor_filter(equalized)

    # Apply Morphological Operations to remove noise and fill holes in the image
    morphed_image = apply_morphology(gabor_filtered)

    # Fast Non-Local Means Denoising for removing noise from the image
    denoised = cv.fastNlMeansDenoising(morphed_image, None, 15)

    # Frangi filter for extracting the vessels from the image and normalizing the image
    vessels = normalize(frangi(denoised), 255).astype(np.uint8)
    _, thresh = cv.threshold(vessels, 0, 255, cv.THRESH_BINARY)
    small_removed = remove_small_elements(thresh, 800).astype(np.uint8)
    end_result = remove_border(color_image, small_removed)

    return end_result


# Check if the image is binary (0 or 255)
def is_binary(image: np.ndarray) -> bool:
    return ((image == 0) | (image == 255)).all()


# Compare the predicted image with the ground truth image
def compare(image: np.ndarray, truth: np.ndarray) -> dict:
    if not is_binary(image):
        raise Exception("Image is not binary (0 or 255)")

    if not is_binary(truth):
        raise Exception("Ground truth is not binary (0 or 255")

    image = image.flatten()
    truth = truth.flatten()

    report = classification_report(truth, image, output_dict=True)

    accuracy = round(
        accuracy_score(truth, image), 2
    )  # accuracy is the ratio of correctly predicted observation to the total observations
    sensitivity = round(
        report["255"]["recall"], 2
    )  # sensitivity is the ratio of correctly predicted positive observations to the all observations in actual class
    specificity = round(
        report["0"]["recall"], 2
    )  # specificity is the ratio of correctly predicted negative observations to the all observations in actual class

    weighted_result = sensitivity_specificity_support(
        truth, image, average="weighted"
    )  # weighted sensitivity and specificity

    weight_sensitivity = round(
        weighted_result[0], 2
    )  # weighted sensitivity is the ratio of correctly predicted positive observations to the all observations in actual class
    weight_specificity = round(weighted_result[1], 2)
    matrix = confusion_matrix(truth, image)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("output/confusion_matrix.png")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "weight_sensitivity": weight_sensitivity,
        "weight_specificity": weight_specificity,
        "matrix": matrix.flatten(),
    }


def visualise(image: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted_vessels = (
        image == 255
    )  # predicted vessels are the pixels with value 255 in the predicted image
    true_vessels = (
        truth == 255
    )  # true vessels are the pixels with value 255 in the ground truth image

    predicted_correct = predicted_vessels & true_vessels  # predicted correctly

    green_predicted = np.zeros(
        (*truth.shape, 3), dtype=np.uint8
    )  # create an empty image with the same shape as the ground truth image and 3 channels
    # set the true vessels to green color and the predicted correctly to green color as well and white for the rest - to visualize the comparison
    green_predicted[true_vessels, :] = 255  # set the true vessels to green color

    green_predicted[predicted_correct] = [
        0,
        255,
        0,
    ]  # set the predicted correctly to green color

    return green_predicted  # return the visualized image


# Streamlit interface for uploading and visualizing images
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

    # Display the confusion matrix
    st.image("output/confusion_matrix.png")

    # Delete the confusion matrix file
    os.remove("output/confusion_matrix.png")


# if __name__ == "__main__":
#     img = process_input(IMAGE_PATH)  # process the input image
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
