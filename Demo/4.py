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
dim_limit = 1920
max_dim = max(image.shape)
if max_dim > dim_limit:
    resize_scale = dim_limit / max_dim
    image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)

# Hiển thị lên màn hình
cv2.imshow('Input', image)
cv2.waitKey()
# Making Copy of original image.
orig_img = image.copy()

# Đọc ảnh thang màu xám
gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

# Áp dụng ngưỡng hóa thích ứng sử dụng phương pháp trung bình và kích thước khối là 9
adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

# Hiển thị ảnh ngưỡng hóa thích ứng
cv2.imshow('Ảnh ngưỡng hóa thích ứng', adaptive_thresh)
cv2.waitKey(0)
