import cv2
import numpy as np
import matplotlib.pyplot as plt



#1 cụm
# Tạo ma trận ngẫu nhiên kích thước 8x8 với giá trị pixel từ 0 đến 255
random_image_X1 = np.array([[52, 55, 61, 66, 70, 61, 64, 73], [63, 59, 55, 90, 109, 85, 69, 72], [62, 59, 68, 113, 144, 104, 66, 73], [63, 58, 71, 122, 154, 106, 70, 69], [
    67, 61, 68, 104, 126, 88, 68, 70], [79, 65, 60, 70, 77, 68, 58, 75], [85, 71, 64, 59, 55, 61, 65, 83], [87, 79, 69, 68, 65, 76, 78, 94]],dtype=np.uint8)
# Tính toán histogram
histogram1, bin_edges1 = np.histogram(random_image_X1, bins=256, range=(0, 256))

# Vẽ biểu đồ histogram
plt.figure()
plt.title("Grayscale Histogram of X1 Image Matrix")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
# axs[0].plot(bin_edges1[0:-1], histogram1)
plt.plot(bin_edges1[0:-1], histogram1)
plt.show()


#1 cụm
# # Tạo ma trận ngẫu nhiên kích thước 8x8 với giá trị pixel từ 0 đến 255
# random_image_X2 = np.array([[19, 17, 2, 1, 1, 2, 2, 1], [18, 19, 19, 17, 1, 1, 1, 1], [17, 18, 19, 17, 1, 2, 1, 1], [18, 19, 19, 19, 19, 1, 1, 2], [
#                             18, 19, 19, 18, 17, 2, 3, 3], [19, 19, 19, 18, 18, 2, 2, 1], [19, 19, 18, 18, 17, 1, 2, 1], [18, 19, 18, 17, 3, 1, 1, 3]],dtype=np.uint8)
# # Tính toán histogram
# histogram2, bin_edges2 = np.histogram(random_image_X2, bins=256, range=(0, 20))

# # Vẽ biểu đồ histogram
# plt.figure()
# plt.title("Grayscale Histogram of X2 Image Matrix")
# plt.xlabel("Pixel value")
# plt.ylabel("Frequency")
# plt.xlim([0, 20])
# #axs[1].plot(bin_edges2[0:-1], histogram2)
# plt.plot(bin_edges2[0:-1], histogram2)
# plt.show()
