from removebg import RemoveBg
import base64
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


from removebg import RemoveBg

rmbg = RemoveBg("61MqkgZ1EUqTAaQNHPTMnS89", "error.log")
rmbg.remove_background_from_img_file(input_image)