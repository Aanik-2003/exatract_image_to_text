# import all required libraries
import cv2
import numpy as np
import easyocr

# initialize easyocr 
reader = easyocr.Reader(['en'])

# load and preprocess image
image_path = 'C:\\Users\\user\\OneDrive\\Desktop\\image_analysis\\img.jpg'
image = cv2.imread(image_path)

# convert image to greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply different parameters for better results (also known as parameter tuning)
thresh_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# remove noise from the image
denoised_image = cv2.GaussianBlur(thresh_image, (3, 3), 0)

# dilate image for enhancing feature of an image
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(denoised_image, kernel, iterations=2)

# extract text from image
result = reader.readtext(dilated_image, detail=0, paragraph=False)

# print extracted result from image
print(result)
