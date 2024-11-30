import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Hàm để áp dụng bộ lọc trung vị với kernel size
def apply_median_filter():
    global img, img_label, is_gray
    
    # Lấy giá trị kernel từ slider
    kernel_size = slider.get()
    if kernel_size % 2 == 0:  # Kernel size phải là số lẻ
        kernel_size += 1
    
    # Áp dụng bộ lọc trung vị
    filtered_image = cv2.medianBlur(img, kernel_size)
    
    # Nếu ảnh là ảnh màu (không phải ảnh xám)
    if not is_gray:
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB để hiển thị

    # Chuyển ảnh từ OpenCV sang PIL để hiển thị với Tkinter
    im_pil = Image.fromarray(filtered_image)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    
    # Cập nhật label hiển thị ảnh
    img_label.configure(image=imgtk)
    img_label.image = imgtk

# Hàm để tải ảnh
def load_image():
    global img, img_label, is_gray
    
    # Chọn ảnh từ file
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        
        # Kiểm tra xem ảnh có phải là ảnh xám không (1 kênh)
        if len(img.shape) == 2:
            is_gray = True
            img_rgb = img  # Nếu là ảnh xám thì không cần chuyển đổi
        else:
            is_gray = False
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Nếu là ảnh màu, chuyển BGR sang RGB
        
        # Hiển thị ảnh ban đầu
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        
        # Cập nhật label hiển thị ảnh ban đầu
        img_label.configure(image=imgtk)
        img_label.image = imgtk
        
        # Áp dụng bộ lọc lần đầu tiên với kernel mặc định
        apply_median_filter()

# Tạo cửa sổ Tkinter
root = Tk()
root.title("Median Filter với Kernel Size Điều Chỉnh")

# Label để hiển thị ảnh
img_label = Label(root)
img_label.pack()

# Tạo slider để điều chỉnh kernel size
slider = Scale(root, from_=1, to=21, orient=HORIZONTAL, label="Kernel Size", command=lambda x: apply_median_filter())
slider.pack()

# Nút để tải ảnh
load_button = Button(root, text="Tải ảnh", command=load_image)
load_button.pack()

# Biến để xác định ảnh có phải là ảnh xám hay không
is_gray = False

# Bắt đầu chương trình
root.mainloop()
