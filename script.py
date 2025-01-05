import cv2
import numpy as np
import easyocr
from pathlib import Path
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

# your path to the captcha images
data_dir = Path("C:\\Users\\user\\OneDrive\\Desktop\\captcha-dataset")

def display_image(img, title="Image"):
    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=2)  

    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(thresh_image, -1, kernel_sharpening)

    denoised_image = cv2.GaussianBlur(sharpened_image, (7, 7), 0)

    kernel = np.ones((1, 1), np.uint8)

    opened_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)

    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    dilated_image = cv2.dilate(closed_image, kernel, iterations=1)

    return dilated_image  

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)

    result = reader.readtext(preprocessed_image, detail=0, paragraph=False, adjust_contrast=True)

    return result, preprocessed_image

def process_and_display_captcha_images(data_dir):
    captcha_texts = {}

    for image_path in data_dir.glob("*.jpg"):  #
        print(f"Actual text: {image_path.name}")

        img = cv2.imread(str(image_path))

        extracted_text, preprocessed_image = extract_text_from_image(img)

        captcha_texts[image_path.name] = " ".join(extracted_text)
        print(f"Extracted Text: {extracted_text}")

        # display_image(preprocessed_image, title=f"Preprocessed Image: {image_path.name}")

    return captcha_texts

if __name__ == "__main__":
    captcha_results = process_and_display_captcha_images(data_dir)