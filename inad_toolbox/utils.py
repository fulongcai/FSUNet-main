import cv2, os, numpy as np

def load_img_as_gray(img_path):
    img = cv2.imread(os.fspath(img_path), cv2.IMREAD_UNCHANGED)  # BGR
    if len(img.shape) != 2:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img is None:
        raise ValueError("File corrupted {}".format(img_path))
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    return img
