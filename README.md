# CIFAR-100 Image Classification Project

## 🚀 Overview
This project implements a **Custom ResNet** model and an optional **CNN model** to classify images from the CIFAR-100 dataset into 100 distinct categories. It incorporates a robust training pipeline, logging, and evaluation framework, making it a great starting point for deep learning enthusiasts to explore image classification techniques.

## 🧩 Key Features
- **Custom ResNet & CNN Models**:
   - Custom-built ResNet inspired by ResNet-18.
   - An alternative CNN model with advanced pooling and activation functions.
   
- **Training Pipeline**:
   - Early stopping and checkpointing support.
   - Advanced techniques such as label smoothing, gradient clipping, and learning rate scheduling.
   
- **Dataset Augmentation**:
   - Includes random cropping, horizontal flipping, color jittering, and normalization for CIFAR-100.
   
- **Evaluation Metrics**:
   - Track training/validation accuracy, loss, and top-k predictions.

- **Visualization**:
   - TensorBoard integration for detailed performance tracking.
   - Augmented image preview saved as PNG.

---

## 📂 Project Structure
```plaintext
CIFAR100-ResNet/
├── checkpoints/          # Directory for saved model weights and training checkpoints.
├── data/                 # CIFAR-100 dataset storage.
├── models/               # Custom model architectures.
│   ├── custom_CNN.py     # Custom CNN architecture.
│   └── custom_ResNet.py  # Custom ResNet architecture.
├── training/             # Training utilities.
│   ├── save_load.py      # Functions to save/load checkpoints.
│   ├── train.py          # Training pipeline.
│   └── early_stopping.py # Early stopping utility.
├── metrics/              # Metrics and loss computations.
│   ├── compute_accuracy.py
│   └── compute_validation_loss.py
├── inference/            # Inference utilities.
│   ├── predict.py        # Script for running inference on new images.
│   └── preprocess.py     # Preprocessing utilities for inference.
├── gui/                  # Optional GUI for inference.
│   └── image_classifier_gui.py
├── configuration/        # Project configurations.
│   └── config_1.py       # Main config file for hyperparameters.
├── logs/                 # TensorBoard logs directory.
├── requirements.txt      # Dependencies.
├── README.md             # Project documentation.
├── main.py               # Main script to run training and evaluation.
├── .gitignore            # Ignore unnecessary files for version control.
```

## 🔧 Getting Started

### 1️⃣ Prerequisites
- Python 3.8+
- pip or conda for managing dependencies.
- GPU-enabled machine (optional but recommended for faster training).

### 2️⃣ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/maxhartml/CIFAR100-ResNet.git
cd CIFAR100-ResNet
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

## 🏋️ Training the Model

### Training Workflow
1. Configure hyperparameters in `configuration/config_1.py`:

```python
BATCH_SIZE = 512
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.6
WEIGHT_DECAY = 1e-4
```

2. Run the training script:

```bash
python main.py
```

3. Monitor progress using TensorBoard:

```bash
tensorboard --logdir ./logs
```

### Key Features in Training:
- **Checkpointing**: Automatically saves model weights to `checkpoints/` every few epochs.
- **Early Stopping**: Stops training if the validation loss doesn’t improve for a specified number of epochs.
- **Augmentation Preview**: Visualize augmented images by enabling `show_augmented_samples()` in `datasets/cifar100_loader.py`.

## 📊 Evaluation

Evaluate your model on the CIFAR-100 test set:
1. After training, the model’s performance is automatically evaluated on the test dataset.
2. Check the test accuracy in the terminal logs:

```plaintext
[INFO] Test Accuracy: 78.34%
```

## 📦 Deployment Options

### 1️⃣ Local Inference GUI

Use the optional Tkinter GUI for local inference:
1. Enable the GUI section in `main.py`:

```python
root = tk.Tk()
app = ImageClassifierGUI(root, model, classes)
root.mainloop()
```

2. Run the script:

```bash
python main.py
```

3. Upload images and view predictions via the GUI.

### 2️⃣ API Deployment

Deploy your model using FastAPI:
1. Set up the API structure under `CIFAR100-Web-App/` as described here.
2. Serve predictions via API endpoints.

## 🛠️ Customization

### Modify the Model Architecture
1. **Custom ResNet**: Edit `models/custom_ResNet.py` to add/remove layers or tweak hyperparameters.
2. **Custom CNN**: Modify `models/custom_CNN.py` to experiment with different pooling or activation strategies.

### Hyperparameter Tuning

Adjust configurations in `configuration/config_1.py`:
- Learning rate schedules
- Regularization (dropout, weight decay)
- Batch size and augmentation techniques

## ✨ Results

| Metric            | Value   |
|-------------------|---------|
| Train Accuracy    | ~99.8%  |
| Validation Accuracy | ~79%   |
| Test Accuracy     | ~78%    |

## 🧪 Augmentation Techniques

| Augmentation     | Description                        |
|------------------|------------------------------------|
| Random Crop      | Adds random cropping for variety.  |
| Horizontal Flip  | Simulates natural flipping.        |
| Colour Jitter    | Adjusts brightness, contrast, etc. |
| Normalization    | Scales pixel values to [-1, 1].    |

Example of augmented images saved as `augmented_samples.png`:

## 🧑‍💻 Contributing

We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a new branch:

```bash
git checkout -b feature/new-feature
```

3. Submit a pull request with your changes.

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more details.

## 📬 Contact

For questions or feedback, reach out to:
- **Name**: Max Hart
- **Email**: maxhart.ml.ai@gmail.com

Thank you for exploring the CIFAR-100 Image Classification Project! 🎉