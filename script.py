import cv2
import easyocr
import numpy as np
from pathlib import Path

reader = easyocr.Reader(['en'])

data_dir = Path("C:\\Users\\user\\OneDrive\\Desktop\\captcha-dataset")

def preprocess_image(image, block_size, C, kernel_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist_eq_image = cv2.equalizeHist(gray)

    thresh_image = cv2.adaptiveThreshold(
        hist_eq_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=block_size, C=C)

    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(thresh_image, -1, kernel_sharpening)

    denoised_image = cv2.GaussianBlur(sharpened_image, (kernel_size, kernel_size), 0)

    kernel = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    dilated_image = cv2.dilate(closed_image, kernel, iterations=1)

    return dilated_image

def extract_text_from_image(image, block_size, C, kernel_size):
    preprocessed_image = preprocess_image(image, block_size, C, kernel_size)
    result = reader.readtext(preprocessed_image, detail=0, paragraph=False, adjust_contrast=True, text_threshold=0.3)
    return result

def evaluate_params(data_dir, block_sizes, Cs, kernel_sizes, max_retries=10):
    captcha_texts = {}
    matches = 0
    total = 0
    successful_params = {}

    for image_path in data_dir.glob("*.jpg"):
        img = cv2.imread(str(image_path))

        expected_label = image_path.stem.split('.')[0]

        match_found = False
        retries = 0

        while not match_found and retries < max_retries:  
            retries += 1
            extracted_text = None

            for block_size in block_sizes:
                for C in Cs:
                    for kernel_size in kernel_sizes:
                        extracted_text = extract_text_from_image(img, block_size, C, kernel_size)

                        if extracted_text and expected_label == extracted_text[0]:
                            match_found = True
                            captcha_texts[image_path.name] = " ".join(extracted_text)
                            successful_params[image_path.name] = (block_size, C, kernel_size)  
                            print(f"Actual text: {expected_label}, Extracted text: {extracted_text[0]}")
                            matches += 1
                            break  
                    if match_found:
                        break  
                if match_found:
                    break  

            if not match_found:
                print(f"No match found for {image_path.name} after {retries} retries, trying different parameters...")

        if not match_found:
            print(f"Failed to extract text for {image_path.name} after {max_retries} retries.")

        total += 1

    accuracy = (matches / total) * 100 if total > 0 else 0
    print(f"\nEvaluation completed. Match accuracy: {accuracy:.2f}%")
    return captcha_texts, successful_params

block_sizes = [11, 15, 17, 19, 21]  
Cs = [1, 2, 3, 4]  
kernel_sizes = [3, 5, 7, 9]  

captcha_results, successful_parameters = evaluate_params(data_dir, block_sizes, Cs, kernel_sizes)

print("\nParameters that worked for each image:")
for image_name, params in successful_parameters.items():
    print(f"{image_name}: Block Size = {params[0]}, C = {params[1]}, Kernel Size = {params[2]}")
