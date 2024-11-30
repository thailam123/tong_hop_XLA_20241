import cv2

# Đọc ảnh
image = cv2.imread("../Image_for_TeamSV/femme.png",cv2.IMREAD_GRAYSCALE)


# Hiển thị ảnh
cv2.imshow("Image", image)

# Chờ đến khi nhấn phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()