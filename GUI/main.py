import tkinter as tk

root = tk.Tk()

label = tk.Label(root,text="Hello world!")
label.pack(padx=20,pady=20)

textbox = tk.Text(root,font=('Arial',16))
textbox.pack()

root.mainloop()

