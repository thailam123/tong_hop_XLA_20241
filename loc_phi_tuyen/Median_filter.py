import cv2

def main():

    # Đọc ảnh đa mức xám
    image = cv2.imread('../Image_for_TeamSV/bagues-noise.jpg', cv2.IMREAD_GRAYSCALE)
    
    median_filter(image,9)

def median_filter(image,ksize=3):
    # Áp dụng bộ lọc trung vị
    blurred_image = cv2.medianBlur(image, ksize=ksize)  # ksize là kích thước kernel, phải là số lẻ

    # Hiển thị kết quả
    cv2.imshow('Original Image', image)
    cv2.imshow('Median Blurred Image', blurred_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()