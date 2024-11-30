import numpy as np
from scipy.ndimage import convolve

image_X2 = np.array([[19, 17, 2, 1, 1, 2, 2, 1], [18, 19, 19, 17, 1, 1, 1, 1], [17, 18, 19, 17, 1, 2, 1, 1], [18, 19, 19, 19, 19, 1, 1, 2], [
    18, 19, 19, 18, 17, 2, 3, 3], [19, 19, 19, 18, 18, 2, 2, 1], [19, 19, 18, 18, 17, 1, 2, 1], [18, 19, 18, 17, 3, 1, 1, 3]], dtype=np.uint8)


# Bộ lọc sharpening 3x3
sharpening_filter = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# Thực hiện phép tích chập với bộ lọc sharpening
result = convolve(image_X2, sharpening_filter, mode='constant', cval=0)

print("Kết quả sau khi áp dụng sharpening filter:")
print(result)

