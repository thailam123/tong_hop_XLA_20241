import numpy as np
from scipy.ndimage import convolve
# import cv2

# # Đọc ảnh
# image = cv2.imread("../Image_for_TeamSV/bagues-noise.jpg",cv2.IMREAD_GRAYSCALE)

image_X2 = np.array([[19, 17, 2, 1, 1, 2, 2, 1], [18, 19, 19, 17, 1, 1, 1, 1], [17, 18, 19, 17, 1, 2, 1, 1], [18, 19, 19, 19, 19, 1, 1, 2], [
    18, 19, 19, 18, 17, 2, 3, 3], [19, 19, 19, 18, 18, 2, 2, 1], [19, 19, 18, 18, 17, 1, 2, 1], [18, 19, 18, 17, 3, 1, 1, 3]], dtype=np.uint8)

# Bộ lọc Gaussian 3x3 với sigma = 1
gaussian_filter_3x3 = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16

# Thực hiện tích chập
result = convolve(image_X2, gaussian_filter_3x3, mode='constant', cval=0)

print("Kết quả sau khi áp dụng Gaussian filtering với bộ lọc 3x3:")
print(result)

# # Hiển thị ảnh
# cv2.imshow("Image", result)
# cv2.imshow("Image0", image)

# # Chờ đến khi nhấn phím bất kỳ để đóng cửa sổ
# cv2.waitKey(0)
# cv2.destroyAllWindows()