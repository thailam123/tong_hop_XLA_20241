import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt

iname = 'Image_for_TeamSV/femme.png'
#read image
img = cv2.imread(iname)


resized = cv2.resize(img, (50, 200))
cv2.imwrite("test.jpg", resized)