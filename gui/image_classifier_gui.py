import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from inference.predict import predict_image

class ImageClassifierGUI:
    """
    A GUI application for image classification using the CIFAR-100 dataset.

    Features:
        - Allows users to upload an image via a file dialog.
        - Displays the uploaded image in the GUI.
        - Predicts the class of the uploaded image using a trained model.
        - Displays the prediction result on the GUI.
    """

    def __init__(self, root, model, device, classes):
        """
        Initialize the GUI with a root window, model, device, and class labels.

        Args:
            root (tk.Tk): The root window for the GUI.
            model (torch.nn.Module): The trained model for prediction.
            device (torch.device): The device (CPU/GPU) to run the inference on.
            classes (list): The list of class labels for CIFAR-100.
        """
        self.root = root
        self.model = model
        self.device = device
        self.classes = classes

        # GUI Configuration
        self.root.title("CIFAR-100 Image Classifier")
        self.root.geometry("600x700")  # Set fixed window size
        self.root.resizable(False, False)  # Disable resizing
        self.root.configure(bg="#2e3b4e")  # Dark theme background

        # Add Title
        self.title_label = tk.Label(
            root,
            text="CIFAR-100 Image Classifier",
            font=("Helvetica", 24, "bold"),
            fg="#ffffff",
            bg="#2e3b4e",
        )
        self.title_label.pack(pady=20)

        # Add Canvas for Image Display
        self.canvas = tk.Canvas(
            root,
            width=400,
            height=400,
            bg="#d3d3d3",
            highlightthickness=2,
            highlightbackground="#4e5d6c",
        )
        self.canvas.pack(pady=20)

        # Add Prediction Result Label
        self.label_result = tk.Label(
            root,
            text="Prediction: None",
            font=("Helvetica", 16, "bold"),
            fg="#f5a623",
            bg="#2e3b4e",
        )
        self.label_result.pack(pady=10)

        # Add Upload Image Button
        self.button_select = tk.Button(
            root,
            text="Upload Image",
            font=("Helvetica", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            command=self.select_image,  # Attach function to button
        )
        self.button_select.pack(pady=20)

        # Footer Information
        self.footer_label = tk.Label(
            root,
            text="Powered by PyTorch and CIFAR-100 Dataset",
            font=("Helvetica", 10),
            fg="#ffffff",
            bg="#2e3b4e",
        )
        self.footer_label.pack(side="bottom", pady=10)

    def select_image(self):
        """
        Opens a file dialog to select an image file.

        - Displays the selected image on the canvas.
        - Predicts and displays the class of the selected image.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:  # Check if a file was selected
            self.display_image(file_path)  # Display the selected image
            self.predict_and_display(file_path)  # Predict and display the result

    def display_image(self, image_path):
        """
        Displays the selected image on the canvas.

        Args:
            image_path (str): The path to the image file.
        """
        try:
            # Open and resize the image to fit the canvas
            img = Image.open(image_path).resize((400, 400))
            self.img_tk = ImageTk.PhotoImage(img)  # Convert to ImageTk format
            self.canvas.create_image(200, 200, image=self.img_tk, anchor=tk.CENTER)
        except Exception as e:
            # Display error in case of failure
            self.label_result.config(text="Error displaying image")
            print(f"[ERROR] Failed to display image: {e}")

    def predict_and_display(self, image_path):
        """
        Predicts the class of the uploaded image and displays the result.

        Args:
            image_path (str): The path to the image file.
        """
        try:
            # Use the predict_image function for inference
            predicted_label = predict_image(self.model, image_path, self.device, self.classes)
            # Display the predicted label on the result label
            self.label_result.config(text=f"Prediction: {predicted_label}")
        except Exception as e:
            # Handle prediction errors
            self.label_result.config(text="Error making prediction")
            print(f"[ERROR] Prediction failed: {e}")