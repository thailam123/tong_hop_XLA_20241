import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage import morphology

def load_binary_image():
    filepath = filedialog.askopenfilename()
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Convert to binary image (0 or 255)
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img_binary

def apply_median_filter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

def apply_max_filter(img, kernel_size):
    return cv2.dilate(img, np.ones((kernel_size, kernel_size)))

def apply_min_filter(img, kernel_size):
    return cv2.erode(img, np.ones((kernel_size, kernel_size)))

def apply_erosion(img, selem_size):
    selem = morphology.disk(selem_size)
    return morphology.erosion(img, selem)

def apply_dilation(img, selem_size):
    selem = morphology.disk(selem_size)
    return morphology.dilation(img, selem)

def apply_opening(img, selem_size):
    selem = morphology.disk(selem_size)
    return morphology.opening(img, selem)

def apply_closing(img, selem_size):
    selem = morphology.disk(selem_size)
    return morphology.closing(img, selem)

def update_image(func, param):
    global img, img_display
    if func == 'median':
        img_display = apply_median_filter(img, param)
    elif func == 'max':
        img_display = apply_max_filter(img, param)
    elif func == 'min':
        img_display = apply_min_filter(img, param)
    elif func == 'erosion':
        img_display = apply_erosion(img, param)
    elif func == 'dilation':
        img_display = apply_dilation(img, param)
    elif func == 'opening':
        img_display = apply_opening(img, param)
    elif func == 'closing':
        img_display = apply_closing(img, param)

    # Update the image in the Tkinter Label
    # Ensure to convert to 0-255 for display
    img_display_tk = ImageTk.PhotoImage(image=Image.fromarray(img_display.astype(np.uint8) * 255))
    label.config(image=img_display_tk)
    label.image = img_display_tk  # Keep a reference

def on_slider_change(val):
    update_image(selected_function.get(), int(val))

def load_and_display_image():
    global img
    img = load_binary_image()
    update_image(selected_function.get(), slider.get())

# Create the main window
root = Tk()
root.title("Morphological Operations on Binary Image")

img = None
img_display = None

# Load Image Button
Button(root, text='Load Binary Image', command=load_and_display_image).pack()

# Function Selection
selected_function = StringVar()
selected_function.set("erosion")

Label(root, text='Select Morphological Operation').pack()
Radiobutton(root, text='Erosion', variable=selected_function, value='erosion').pack()
Radiobutton(root, text='Dilation', variable=selected_function, value='dilation').pack()
Radiobutton(root, text='Opening', variable=selected_function, value='opening').pack()
Radiobutton(root, text='Closing', variable=selected_function, value='closing').pack()

# Slider for changing kernel size
slider = Scale(root, from_=1, to=20, orient=HORIZONTAL, command=on_slider_change)
slider.pack()

# Label for displaying the image
label = Label(root)
label.pack()

root.mainloop()
