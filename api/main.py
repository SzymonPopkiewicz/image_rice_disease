import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import os

model = tf.keras.models.load_model("images/model_rdy/first_model.keras")
classes = ['Bacterialblight', 'Brownspot', 'Leafsmut']


def read_image(filepath):
    image = Image.open(filepath)
    return image

def upload_image():

    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg; *.jpeg; *.png")])
    if filepath:
        try:
            image = read_image(filepath)
            img_array = np.expand_dims(image, 0)
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])
            result_label.config(text=f"Predicted Class: {predicted_class}\nConfidence: {confidence*100:.2f}%")
            image.thumbnail((256, 256)) 
            photo = ImageTk.PhotoImage(image)
            img_label.config(image=photo)
            img_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x360")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()
result_label = tk.Label(root, text="")
result_label.pack(pady=10)
root.mainloop()