import cv2
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg

from PIL import Image

def select_file():
    """Open a file dialog and return the selected file path."""
    file_path = sg.popup_get_file("Select file", no_window=True)
    return file_path

def preprocess_image(img_path):
    """Preprocess the input image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Resize image if necessary
    max_dim = max(img.shape)
    dim_limit = 1080
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    return img

def remove_text(img):
    """Remove text from the document."""
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    return img

def extract_document(img):
    """Extract the document from the background."""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img

def edge_detection(img):
    """Perform edge detection on the preprocessed image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(gray, 100, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return canny

def find_document_contour(canny_img):
    """Find contours for the detected edges."""
    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    return page

def order_corner_points(pts):
    """Order corner points of the document."""
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype('int').tolist()

def perspective_transform(img, corners):
    """Apply perspective transform to the document."""
    (tl, tr, br, bl) = corners
    maxWidth = max(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)),
                   np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)))
    maxHeight = max(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)),
                    np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(img, np.float32(homography), (int(maxWidth), int(maxHeight)), flags=cv2.INTER_LINEAR)
    return final

def main():
    sg.theme('LightGrey1')

    file_path = select_file()

    if file_path:
        print(f"Selected file: {file_path}")
        original_image = preprocess_image(file_path)
        img_copy = original_image.copy()

        img_no_text = remove_text(img_copy)

        extracted_doc = extract_document(img_no_text)

        edges = edge_detection(extracted_doc)

        contours = find_document_contour(edges)

        for c in contours:
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                break

        con = np.zeros_like(extracted_doc)
        cv2.drawContours(con, [c], -1, (0, 255, 255), 3)
        cv2.drawContours(con, corners, -1, (0, 255, 0), 10)

        corners = sorted(np.concatenate(corners).tolist())
        corners = order_corner_points(corners)

        final_image = perspective_transform(original_image, corners)

        plt.figure(figsize=[20, 10])
        plt.subplot(121)
        plt.imshow(original_image[:, :, ::-1])
        plt.axis('off')
        plt.title("Original image")
        plt.subplot(122)
        plt.imshow(final_image[:, :, ::-1])
        plt.axis('off')
        plt.title("Scanned Form")
        plt.show()
    else:
        print("File selection canceled.")


if __name__ == "__main__":
    main()
