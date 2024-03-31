import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from keras.models import load_model
import os
from tkinter import ttk


class ImageForgeryCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ELA Image Forgery Checker")
        self.root.geometry("600x400")  # Larger window size

        # Style for widgets
        self.style = tk.ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))
        self.style.configure("TLabel", font=("Helvetica", 12))
        self.style.configure("TFrame", background="lightgray")

        self.model_path = "Re_Traind_Models/ELA_model.h5"

        self.label = tk.Label(self.root, text="Choose an image to check:")
        self.label.pack(pady=10)

        self.btn_browse = tk.Button(self.root, text="Browse", command=self.browse_image)
        self.btn_browse.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        try:
            label, prob = self.test_image_with_ela(image_path, self.model_path)
            messagebox.showinfo(
                "Prediction",
                f"The image is {label} with a probability of {prob:.2f}",
                parent=self.root,
            )  # Show messagebox inside the main window
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred: {str(e)}", parent=self.root
            )

    def test_image_with_ela(self, image_path, model_path):

        # loading Model
        model = load_model(model_path)
        # Read image
        image_saved_path = image_path.split(".")[0] + ".saved.jpg"

        # calculate ELA
        image = Image.open(image_path).convert("RGB")
        image.save(image_saved_path, "JPEG", quality=90)
        saved_image = Image.open(image_saved_path)
        ela = ImageChops.difference(image, saved_image)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_im = ImageEnhance.Brightness(ela).enhance(scale)

        # prepare image for testing
        image = np.array(ela_im.resize((128, 128))).flatten() / 255.0
        image = image.reshape(-1, 128, 128, 3)
        # prediction
        prob = model.predict(image)[0]
        idx = np.argmax(prob)
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]

        label = "Forged" if pred == 1 else "Not Forged"
        return label, prob[idx]


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageForgeryCheckerApp(root)
    root.mainloop()
