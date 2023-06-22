import cv2


def read_image(image_directory):
    img = cv2.imread(image_directory, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    return img
