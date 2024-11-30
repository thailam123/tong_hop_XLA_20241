import numpy as np
from scipy.ndimage import convolve
from tkinter import filedialog
import cv2
from tkinter import Tk, Button
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
import inspect


def loc_mean_gray_3x3(image1):
    # Đọc ảnh
    image = image1

    # Bộ lọc trung bình 3x3
    mean_filter_3x3 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                # Chia cho 9 để đảm bảo tổng bằng 1
                                [1, 1, 1]]) / 9

    # Thực hiện tích chập
    result = convolve(image, mean_filter_3x3, mode='constant', cval=1)

    print(result)


def loc_min(random_image_X2):
    # Đọc ảnh (grayscale để đơn giản)
    image = random_image_X2

    # Kích thước kernel
    kernel_size = 3

    # Tạo kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Áp dụng lọc Min
    # borderType=cv2.BORDER_CONSTANT, borderValue=255
    min_filtered_image = cv2.erode(image, kernel)

    print(image)
    print('\n')
    print(min_filtered_image)


def power_law_transform(random_image_X1, gamma, c=5):
    # Đọc ảnh grayscale
    image = random_image_X1
    # Chuẩn hóa ảnh để giá trị pixel nằm trong khoảng [0, 1]
    normalized_image = image / 255.0

    # Thực hiện biến đổi power-law với công thức S = c * r^gamma
    # Giá trị của transformed_image thường nằm trong khoảng [0,c], với 𝑐 = 5
    transformed_image = c * np.power(normalized_image, gamma)

    # Chuyển lại giá trị pixel về khoảng [0, 255]
    # Bất kỳ giá trị nào nhỏ hơn 0 sẽ được đặt thành 0.
    # Bất kỳ giá trị nào lớn hơn 255 sẽ được đặt thành 255.
    result_image = np.uint8(np.clip(transformed_image * 255, 0, 255))
    print(result_image)


random_image_X2 = np.array([
    [19, 17, 2, 1, 1, 2, 2, 1],
    [18, 19, 19, 17, 1, 1, 1, 1],
    [17, 18, 19, 17, 1, 2, 1, 1],
    [18, 19, 19, 19, 19, 1, 1, 2],
    [18, 19, 19, 18, 17, 2, 3, 3],
    [19, 19, 19, 18, 18, 2, 2, 1],
    [19, 19, 18, 18, 17, 1, 2, 1],
    [18, 19, 18, 17, 3, 1, 1, 3]
], dtype=np.uint8)

random_image_X1 = np.array([[52, 55, 61, 66, 70, 61, 64, 73], [63, 59, 55, 90, 109, 85, 69, 72], [62, 59, 68, 113, 144, 104, 66, 73], [63, 58, 71, 122, 154, 106, 70, 69], [
    67, 61, 68, 104, 126, 88, 68, 70], [79, 65, 60, 70, 77, 68, 58, 75], [85, 71, 64, 59, 55, 61, 65, 83], [87, 79, 69, 68, 65, 76, 78, 94]], dtype=np.uint8)

loc_min(random_image_X2)
