import numpy as np
from scipy.ndimage import convolve

image_X2 = np.array([[19, 17, 2, 1, 1, 2, 2, 1], [18, 19, 19, 17, 1, 1, 1, 1], [17, 18, 19, 17, 1, 2, 1, 1], [18, 19, 19, 19, 19, 1, 1, 2], [
    18, 19, 19, 18, 17, 2, 3, 3], [19, 19, 19, 18, 18, 2, 2, 1], [19, 19, 18, 18, 17, 1, 2, 1], [18, 19, 18, 17, 3, 1, 1, 3]], dtype=np.uint8)

# Bộ lọc mean 3x3
mean_filter = np.ones((3, 3)) / 9

# Thực hiện phép tích chập
result = convolve(image_X2, mean_filter, mode='constant', cval=0)

print("Kết quả sau khi áp dụng lọc trung bình:")
print(result)
