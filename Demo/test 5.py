
import pytesseract
import PIL.Image
import cv2

import PySimpleGUI as sg
from pytesseract import pytesseract


def chon_file():
  """Mở hộp thoại chọn file và trả về đường dẫn file được chọn."""
  file_path = sg.popup_get_file("Chọn file", no_window=True)
  return file_path

file_path = chon_file()

if file_path:
  print(f"Đã chọn file: {file_path}")
else:
  print("Hủy chọn file.")


# Tên file ảnh đầu vào
input_image  = file_path

# Đọc ảnh
image = cv2.imread(input_image)



# Resize image to workable size
dim_limit = 1080
max_dim = max(image.shape)
if max_dim > dim_limit:
    resize_scale = dim_limit / max_dim
    image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)

# Hiển thị lên màn hình
cv2.imshow('Input', image)
cv2.waitKey()
# Making Copy of original image.
orig_img = image.copy()
"""
Page segmentation modes: 
O Orientation and script detection (OSD) only
1 Automatic page segmentation with OSD. ‘
2 Automatic page segmentation, but no OSD, or OCR.
3 Fully automatic page segmentation, but no OSD. (Default)
4 Assume a single column of text of variable sizes.
5 Assume a single uniform block of vertically aligned text.
6 Assume a single uniform block of textJ
7 Treat the image as a single text line.
8 Treat the image as a single word.
9 Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific."""

myconfig = r" --psm 3 --oem 3"
#img = cv2.imread(orig_img)
height, width, _ = orig_img.shape
boxes = pytesseract.image_to_boxes(orig_img, config=myconfig)
for box in boxes.splitlines():
    box = box.split(' ')
    image = cv2.rectangle(image, (int(box[1]), height -int(box[2])), (int(box[3]), height -int(box[4])), (0, 255, 0), 2)


text = pytesseract.image_to_string(orig_img, config=myconfig)

print(text)
#print(boxes)

cv2.imshow('Img', image)
cv2.waitKey(0)
