import numpy as np
from scipy.ndimage import convolve
from tkinter import filedialog
import cv2
import cv2 as cv
from matplotlib import pyplot as plt
from tkinter import Tk, Button
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter


def main():

    Histogram_gray("C:/Users/thaiv/.spyder-py3/xu_ly_anh/Image_for_TeamSV/bagues-noise.jpg")
    



# thay đổi độ tương phản ảnh, hình 3
def Linear_Contrast_Stretching_Transformation(filepath, r1=70, s1=50, r2=140, s2=200):
    def contrast_stretch(image, r1, s1, r2, s2):
        # Create a lookup table for intensity mapping
        #If lut[50] = 100, all pixels with intensity 50 in image will be replaced with 100 in stretched_image.
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            #3 dòng điều kiện này là theo công thức mà có
            if i < r1:
                lut[i] = int((s1 / r1) * i)
            elif r1 <= i <= r2:
                lut[i] = int(((s2 - s1) / (r2 - r1)) * (i - r1) + s1)
            else:
                lut[i] = int(((255 - s2) / (255 - r2)) * (i - r2) + s2)
        # Apply the lookup table to the image
        stretched_image = cv2.LUT(image, lut)
        return stretched_image
    # Load a grayscale image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Apply the linear contrast stretching
    stretched = contrast_stretch(image, r1, s1, r2, s2)

    # Display the original and transformed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Contrast-Stretched Image")
    plt.imshow(stretched, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def Histogram_gray(filepath):
    # Đọc ảnh đa mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Tính histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Cân bằng histogram
    equalized_image = cv2.equalizeHist(image)

    # Tính histogram sau cân bằng
    equalized_hist, _ = np.histogram(equalized_image.flatten(), 256, [0, 256])

    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Ảnh gốc
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Ảnh gốc')
    axes[0, 0].axis('off')

    # Biểu đồ histogram ảnh gốc
    axes[0, 1].bar(range(256), hist, color='gray')
    axes[0, 1].set_title('Histogram Ảnh gốc')

    # Ảnh sau khi cân bằng
    axes[1, 0].imshow(equalized_image, cmap='gray')
    axes[1, 0].set_title('Ảnh sau cân bằng')
    axes[1, 0].axis('off')

    # Biểu đồ histogram ảnh sau cân bằng
    axes[1, 1].bar(range(256), equalized_hist, color='gray')
    axes[1, 1].set_title('Histogram Ảnh sau cân bằng')

    plt.tight_layout()
    plt.show()


def max_color_3x3(filepath):
    # Đọc ảnh màu
    image = cv2.imread(filepath)

    # Tạo kernel 3x3
    kernel = np.ones((3, 3), np.uint8)

    # Áp dụng bộ lọc max (dilate)
    max_filtered_image = cv2.dilate(image, kernel)

    # Hiển thị kết quả
    cv2.imshow("Original Image", image)
    cv2.imshow("Max Filtered Image", max_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def max_gray_3x3(filepath):
    # Đọc ảnh mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Tạo mặt nạ 3x3 cho bộ lọc max
    kernel = np.ones((3, 3), np.uint8)

    # Áp dụng bộ lọc max (sử dụng hàm dilate của OpenCV)
    max_filtered_image = cv2.dilate(image, kernel)

    # Hiển thị kết quả
    cv2.imshow('Original Image', image)
    cv2.imshow('Max Filtered Image', max_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def min_color_3x3(filepath):
    # Đọc ảnh màu
    image = cv2.imread(filepath)

    # Chuyển đổi ảnh sang không gian màu HSV để tách biệt màu và độ sáng
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo một kernel 3x3 cho lọc min
    kernel = np.ones((3, 3), np.uint8)

    # Áp dụng lọc min cho từng kênh HSV
    min_hue = cv2.erode(hsv[:, :, 0], kernel)
    min_saturation = cv2.erode(hsv[:, :, 1], kernel)
    min_value = cv2.erode(hsv[:, :, 2], kernel)

    # Hợp nhất các kênh đã lọc để tạo lại ảnh HSV
    min_hsv = cv2.merge([min_hue, min_saturation, min_value])

    # Chuyển ảnh HSV đã lọc ngược lại sang không gian màu BGR
    min_filtered_image = cv2.cvtColor(min_hsv, cv2.COLOR_HSV2BGR)

    # Hiển thị ảnh gốc và ảnh đã qua lọc min
    cv2.imshow('Original Image', image)
    cv2.imshow('Min Filtered Image', min_filtered_image)

    # Đợi phím bấm và đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def min_gray_3x3(filepath):
    # Đọc ảnh đầu vào dưới dạng ảnh đa mức xám
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Tạo kernel (mặt nạ) 3x3 với tất cả các giá trị là 1
    kernel = np.ones((3, 3), np.uint8)

    # Áp dụng phép lọc min
    min_filtered_img = cv2.erode(img, kernel)

    # Lưu hoặc hiển thị ảnh kết quả
    cv2.imwrite('min_filtered_image.png', min_filtered_img)
    cv2.imshow('Min Filtered Image', min_filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def median_color_3x3(filepath):
    # Đọc ảnh màu
    image = cv2.imread(filepath)

    # Áp dụng bộ lọc trung vị với mặt nạ 3x3
    filtered_image = cv2.medianBlur(image, 3)

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng bộ lọc
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def median_gray_3x3(filepath):
    # Đọc ảnh đa mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Áp dụng bộ lọc trung vị với mặt nạ 3x3
    median_filtered = cv2.medianBlur(image, 3)

    # Hiển thị ảnh gốc và ảnh sau khi lọc
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Ảnh sau khi lọc trung vị 3x3")
    plt.imshow(median_filtered, cmap='gray')

    plt.show()


def sobel_gray_3x3(filepath):
    # Đọc ảnh mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Tính đạo hàm theo hướng x và y với Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # đạo hàm theo x
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # đạo hàm theo y

    # Tính gradient tổng hợp
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Hiển thị ảnh gốc và ảnh sau khi lọc Sobel
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Sobel X")
    plt.imshow(np.abs(sobel_x), cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Sobel Y")
    plt.imshow(np.abs(sobel_y), cmap="gray")

    plt.show()


def sharpening_color_9x9(filepath):
    # Load the color image
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a 9x9 sharpening kernel
    sharpen_kernel = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 81, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

    # Display the original and sharpened images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image)
    plt.title("Sharpened Image")

    plt.show()


def sharpening_color_3x3(filepath):
    # Đọc ảnh màu
    image = cv2.imread(filepath)
    # Đổi màu ảnh từ BGR sang RGB để hiển thị đúng màu
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Định nghĩa kernel sharpening 3x3
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    # Áp dụng bộ lọc
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Hiển thị ảnh gốc và ảnh đã làm sắc nét
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image)
    plt.title("Sharpened Image")
    plt.axis("off")

    plt.show()

# γ là giá trị gamma điều chỉnh. Giá trị γ<1 sẽ làm sáng ảnh, còn γ>1 sẽ làm tối ảnh.
# Power-Law Transformations
def power_law_transform(filepath, gamma, c=5):
    # Đọc ảnh grayscale
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Chuẩn hóa ảnh để giá trị pixel nằm trong khoảng [0, 1]
    normalized_image = image / 255.0

    # Thực hiện biến đổi power-law với công thức S = c * r^gamma
    # Giá trị của transformed_image thường nằm trong khoảng [0,c], với 𝑐 = 5 
    transformed_image = c * np.power(normalized_image, gamma)

    # Chuyển lại giá trị pixel về khoảng [0, 255]
    # Bất kỳ giá trị nào nhỏ hơn 0 sẽ được đặt thành 0.
    # Bất kỳ giá trị nào lớn hơn 255 sẽ được đặt thành 255.
    result_image = np.uint8(np.clip(transformed_image * 255, 0, 255))

    # Hiển thị ảnh gốc và ảnh sau khi biến đổi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Transformed Image (c={c}, gamma={gamma})")
    plt.imshow(result_image, cmap="gray")
    plt.axis("off")

    plt.show()


# chuyển ảnh đa mức xám dương bản sang âm bản
def duong_ban_sang_am_ban(filepath):
    # Đọc ảnh dương bản
    image = cv2.imread(filepath)
    # Chuyển đổi sang âm bản
    negative_image = 255 - image
    cv2.imshow("image result of changing to negative iamge", negative_image)

    # Chờ đến khi nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sharpening_gray_9x9(filepath):
    # Đọc ảnh ở dạng đa mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Tạo kernel làm sắc nét 9x9
    # Kernel này làm sắc nét bằng cách giữ lại giá trị trung tâm lớn và các giá trị xung quanh nhỏ hơn
    sharpening_kernel = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 81, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])

    # Áp dụng bộ lọc làm sắc nét
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    # Hiển thị ảnh gốc và ảnh sau khi làm sắc nét
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Ảnh gốc')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Ảnh làm sắc nét')
    plt.imshow(sharpened_image, cmap='gray')

    plt.show()


def sharpening_gray_3x3(filepath):
    # Đọc ảnh ở dạng đa mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Tạo kernel làm sắc nét 3x3
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    # Áp dụng bộ lọc làm sắc nét
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    # Hiển thị ảnh gốc và ảnh sau khi làm sắc nét
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Ảnh gốc')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Ảnh làm sắc nét')
    plt.imshow(sharpened_image, cmap='gray')

    plt.show()


def mean_color_9x9(filepath):
    # Đọc ảnh
    image = cv2.imread(filepath)  # Đường dẫn tới ảnh của bạn
    # Chuyển đổi sang RGB để hiển thị đúng với Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Áp dụng bộ lọc mean với kernel 9x9
    kernel_size = (9, 9)
    filtered_image = cv2.blur(image_rgb, kernel_size)

    # Hiển thị ảnh gốc và ảnh đã lọc
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Ảnh gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image)
    plt.title('Ảnh sau khi lọc mean (9x9)')

    plt.show()


def mean_color_3x3(filepath):
    # Đọc ảnh
    image = cv2.imread(filepath)  # Đường dẫn tới ảnh của bạn
    # Chuyển đổi sang RGB để hiển thị đúng với Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Áp dụng bộ lọc mean với kernel 3x3
    kernel_size = (3, 3)
    filtered_image = cv2.blur(image_rgb, kernel_size)

    # Hiển thị ảnh gốc và ảnh đã lọc
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Ảnh gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image)
    plt.title('Ảnh sau khi lọc mean (3x3)')

    plt.show()


def mean_gray_9x9(filepath):
    # Đọc ảnh đa mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Áp dụng bộ lọc trung bình với kernel 9x9
    kernel_size = (9, 9)
    filtered_image = cv2.blur(image, kernel_size)

    # Hiển thị ảnh gốc và ảnh sau khi lọc
    cv2.imshow("Original Image", image)
    cv2.imshow("Mean Filtered Image", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loc_Gaussian_color_9x9(filepath):
    # Đọc ảnh màu
    image = cv2.imread(filepath)

    # Áp dụng bộ lọc Gaussian với kernel 9x9
    blurred_image = cv2.GaussianBlur(image, (9, 9), 0)

    # Hiển thị ảnh kết quả
    cv2.imshow('Original Image', image)
    cv2.imshow('Gaussian Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loc_Gaussian_gray_9x9(filepath):
    # Đọc ảnh và chuyển thành ảnh đa mức xám
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Áp dụng bộ lọc Gaussian với kernel 9x9
    blurred_image = cv2.GaussianBlur(image, (9, 9), 0)

    # Hiển thị ảnh kết quả
    cv2.imshow('Original Image', image)
    cv2.imshow('Gaussian Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loc_Gaussian_color_3x3(filepath):
    # Đọc ảnh màu
    image = cv2.imread(filepath)

    # Áp dụng bộ lọc Gaussian với kernel 3x3
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Hiển thị ảnh kết quả
    cv2.imshow('Original Image', image)
    cv2.imshow('Gaussian Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loc_Gaussian_gray_3x3(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Bộ lọc Gaussian 3x3 với sigma = 1
    gaussian_filter_3x3 = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16
    # Thực hiện tích chập
    # mode='constant', cval=0 xử lý edge
    result = convolve(image, gaussian_filter_3x3, mode='constant', cval=0)

    # cv2.imshow("Ảnh gốc", image)
    cv2.imshow("image result of Gaussian filter", result)

    # Chờ đến khi nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("loc_Gaussian_gray: ", result)


def loc_mean_gray_3x3(filepath):
    # Đọc ảnh
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Bộ lọc trung bình 3x3
    mean_filter_3x3 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                # Chia cho 9 để đảm bảo tổng bằng 1
                                [1, 1, 1]]) / 9

    # Thực hiện tích chập
    result = convolve(image, mean_filter_3x3, mode='constant', cval=0)

    # Hiển thị ảnh đã lọc
    cv2.imshow("image result of mean filter", result)

    # Chờ đến khi nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
