import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, Toplevel, Scale, Label, Button
from PIL import Image, ImageTk
    
random_image_X1 = np.array([[52, 55, 61, 66, 70, 61, 64, 73], [63, 59, 55, 90, 109, 85, 69, 72], [62, 59, 68, 113, 144, 104, 66, 73], [63, 58, 71, 122, 154, 106, 70, 69], [
    67, 61, 68, 104, 126, 88, 68, 70], [79, 65, 60, 70, 77, 68, 58, 75], [85, 71, 64, 59, 55, 61, 65, 83], [87, 79, 69, 68, 65, 76, 78, 94]],dtype=np.uint8)

kernel_shape = {'Rectangle': cv2.MORPH_RECT, 'Ellipse': cv2.MORPH_ELLIPSE, 'Cross': cv2.MORPH_CROSS}
chosen_shape = kernel_shape['Cross']

kernel = cv2.getStructuringElement(chosen_shape, (5, 5))

filtered_image = cv2.medianBlur(random_image_X1, 3)

print(random_image_X1)
print('\n')
print(filtered_image)