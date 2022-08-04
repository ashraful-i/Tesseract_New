import numpy as np
from PIL import Image
import pytesseract
import cv2
from pytesseract import Output


# from scipy.ndimage import interpolation as inter

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)


# Scaling of image 300 DPI
def imageResize(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)  # Inter Cubic
    return img


# BGR to GRAY
def bgrtogrey(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# Increase Brightness
def increaseBrightness(img):
    alpha = 1
    beta = 40
    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    return img

    # if dark increase brightness
    if checker == 1:
        img = increaseBrightness(img)  # increase brightness


def threshold(img):
    # Various thresholding method
    img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return img


# Noise reduction
def noise_removal(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def extractinformation(img):
    extractedInformation = pytesseract.image_to_string(img)
    print(extractedInformation)


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
image_to_ocr = cv2.imread('us1.jpg')
processed_img = image_to_ocr
processed_img = imageResize(processed_img)
# processed_img = correct_skew(processed_img)
processed_img = bgrtogrey(processed_img)
# processed_img = threshold(processed_img)
processed_img = noise_removal(processed_img)
processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

## Example 1
cv2.imwrite("temp_img.png", processed_img)
processed_img = Image.open("temp_img.png")

print("Processed Image")
custom_conf1 = ''
text_extract = pytesseract.image_to_data(processed_img, output_type=Output.DICT, config=custom_conf1)
# print("1 : " + text_extract['text'])
# print(text_extract)
for count, item in enumerate(text_extract['conf']):
    if float(item) > 50:
        print("Text: " + text_extract['text'][count], "\n Conf: " + str(item), " left: " + str(text_extract['left'][count]), " top: " + str(text_extract['top'][count]), " width: " + str(text_extract['width'][count]))
print('\n')

print("Processed Image with custom config")
custom_conf1 = r'--psm 11'
text_extract = pytesseract.image_to_data(processed_img, output_type=Output.DICT, config=custom_conf1)
# print(text_extract)

for count, item in enumerate(text_extract['conf']):
    if float(item) > 50:
        print("Text: " + text_extract['text'][count], "\n Conf: " + str(item), " left: " + str(text_extract['left'][count]), " top: " + str(text_extract['top'][count]), " width: " + str(text_extract['width'][count]))
# print("2 : " + text_extract['text'])
print('\n')

print("Original Image with custom config")
custom_conf1 = r'--psm 11'
text_extract = pytesseract.image_to_data(image_to_ocr, output_type=Output.DICT, config=custom_conf1)
# print(text_extract)
for count, item in enumerate(text_extract['conf']):
    if float(item) > 50:
        print("Text: " + text_extract['text'][count], "\n Conf: " + str(item), " left: " + str(text_extract['left'][count]), " top: " + str(text_extract['top'][count]), " width: " + str(text_extract['width'][count]))
# print("3: " + text_extract['text'])
print('\n')

print("Original Image without custom config")
custom_conf1 = ''
text_extract = pytesseract.image_to_data(image_to_ocr, output_type=Output.DICT, config=custom_conf1)
# print(text_extract)
for count, item in enumerate(text_extract['conf']):
    if float(item) > 50:
        print("Text: " + text_extract['text'][count], "\n Conf: " + str(item), " left: " + str(text_extract['left'][count]), " top: " + str(text_extract['top'][count]), " width: " + str(text_extract['width'][count]))
# print("4: " + text_extract['text'])
# custom_conf1 = r'--psm 11'
# custom_conf1='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
print('\n')