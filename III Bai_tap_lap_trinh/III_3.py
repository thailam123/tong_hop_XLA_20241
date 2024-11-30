import cv2
import matplotlib.pyplot as plt

#cụm 1
# # Đọc ảnh
# image = cv2.imread('Image_for_TeamSV/femme.png', cv2.IMREAD_GRAYSCALE)

# # Tính toán histogram
# hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# # Vẽ biểu đồ
# plt.figure(figsize=(10, 6))
# plt.plot(hist, color='black')
# plt.title('Histogram of Grayscale Image')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.xlim([0, 256])
# plt.grid(True)
# plt.show()
#cụm 1

#cụm 2
# Đọc ảnh màu
image = cv2.imread('../Image_for_TeamSV/Fruit.jpg')

# Chuyển đổi không gian màu từ BGR (OpenCV sử dụng) sang RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Tạo danh sách tên các kênh màu và màu tương ứng để vẽ
colors = ('r', 'g', 'b')
channels = ('Red', 'Green', 'Blue')

plt.figure(figsize=(10, 6))

# Vẽ histogram cho mỗi kênh màu
for i, color in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=channels[i])
    plt.xlim([0, 256])

# Thêm tiêu đề và nhãn
plt.title('Histogram of Color Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
#cụm 2