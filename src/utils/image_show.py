import cv2

from utils.read_image import read_image

def image_show(image_directory):
    img = read_image(image_directory)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
