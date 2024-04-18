import cv2
import PySimpleGUI as sg


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
img = cv2.imread(input_image)

# Hiển thị lên màn hình
cv2.imshow('Input', img)
cv2.waitKey()
# Making Copy of original image.
orig_img = img.copy()

# Chuyển ảnh mày thành ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 41, 5)

cv2.imshow('Thresh', thresh)
cv2.imshow('Thresh', adaptive_thresh)
cv2.waitKey()

