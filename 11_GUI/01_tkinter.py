from tkinter import *
from tkinter import messagebox

root = Tk()
root.title("my first GUI")
root.geometry("500x300+100+100")

btn01 = Button(root)  # This is the button widget.
btn01["text"] = "Hello World"  # This is the text that will be displayed on the button.
btn01.pack()  # This is the method that will pack the button widget into the window.


def action(a):  # a是事件对象
    messagebox.showinfo("Message",
                        "Button Clicked")  # This is the method that will display a message box when the button is clicked.
    print("Button Clicked")


btn01.bind("<Button-1>", action)  # Button-1表示鼠标左键点击
# btn01["command"] = action # This is the method that will be called when the button is clicked.

root.mainloop()  # This is the main loop of the application. It will run until the application is closed.
