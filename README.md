# Captcha Solver

A CAPTCHA text recognition project built with OpenCV, SciPy, and TensorFlow/Keras. It preprocesses noisy CAPTCHA images, segments individual characters using morphological filtering and contour analysis, and classifies each extracted `28x28` grayscale character patch using a Convolutional Neural Network (CNN).

## Overview

The repository contains two Google Colab notebooks, a pre-trained Keras model (`model5.h5`), and a character reference file (`character.txt`):

- **Character Segmentation & Inference (`main.ipynb`)**: Preprocesses CAPTCHA images, removes line and circle noise, segments individual characters into `28x28` grayscale images, and decodes the CAPTCHA text using `model5.h5`.
- **CNN Model Training (`Captcha_Solver.ipynb`)**: Defines and trains the `28x28` grayscale CNN classifier on a CSV dataset using data augmentation (`ImageDataGenerator`) and exports the trained model as `model5.h5`.

## Key Features

- **Noise Removal & Preprocessing (`main.ipynb`)**:
  - Applies `8px` replicate border padding, `3x3` Gaussian blur, inverted Otsu thresholding (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`), and `3x3` dilation.
  - Uses morphological erosion and alternating `(5, 1)` and `(1, 5)` median filters (`scipy.ndimage.median_filter`) to remove line noise and weaken circle noise.
  - Filters connected components (`cv2.connectedComponentsWithStats`) in `purify()` to discard small noise below `0.1 * meanArea`, pads regions to square dimensions, and resizes them to `28x28` (`cv2.INTER_AREA`).
- **Adaptive Contour-Based Segmentation (`main.ipynb`)**:
  - Computes `getRatioArea()` (largest contour area divided by total contour area) to dynamically select an erosion kernel (`2x2`, `3x3`, or `5x5`).
  - Filters out small contours (`cv2.contourArea(contour) > 0.05 * meanArea`) and splits wide bounding boxes (`w / h > 1.25`) horizontally into two halves to separate touching characters.
  - Sorts candidate bounding boxes left-to-right and recursively retries segmentation (up to `recursion < 3`) whenever more than 7 candidate regions are detected.
- **CNN Classification (`Captcha_Solver.ipynb`, `model5.h5`)**:
  - Three-stage convolutional architecture (`32`, `64`, and `128` filters) with `BatchNormalization`, strided convolutions (`strides=2`), and `Dropout(0.4)` regularization.
  - Decodes digits `0`–`9` and uppercase letters `A`–`Z` (`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`).

## Model Architecture

The CNN defined in `Captcha_Solver.ipynb` takes `28x28x1` normalized grayscale character images (`imgs / 255`) and is compiled with the `adam` optimizer and `categorical_crossentropy` loss:

- **Stage 1**: `Conv2D(32, 3x3, relu)` -> `BatchNormalization` -> `Conv2D(32, 3x3, relu)` -> `BatchNormalization` -> `Conv2D(32, 5x5, strides=2, padding='same', relu)` -> `BatchNormalization` -> `Dropout(0.4)`
- **Stage 2**: `Conv2D(64, 3x3, relu)` -> `BatchNormalization` -> `Conv2D(64, 3x3, relu)` -> `BatchNormalization` -> `Conv2D(64, 5x5, strides=2, padding='same', relu)` -> `BatchNormalization` -> `Dropout(0.4)`
- **Classifier Head**: `Conv2D(128, 4x4, relu)` -> `BatchNormalization` -> `Flatten` -> `Dropout(0.4)` -> `Dense(62, softmax)`

## Technologies Used

- **Python 3** (Jupyter Notebook / Google Colab)
- **TensorFlow / Keras** (`Sequential`, `load_model`, `Conv2D`, `BatchNormalization`, `Dropout`, `Flatten`, `Dense`, `ImageDataGenerator`, `ModelCheckpoint`, `to_categorical`)
- **OpenCV** (`cv2`)
- **SciPy** (`scipy.ndimage`)
- **NumPy** (`numpy`)
- **pandas** (`pandas`)

## Project Structure

```text
.
├── Captcha_Solver.ipynb   # CNN architecture, CSV data loader, augmentation, and training notebook
├── main.ipynb             # Image preprocessing, character segmentation, and inference notebook
├── model5.h5              # Pre-trained Keras CNN model (~4.14 MB)
├── character.txt          # Reference list of predicted characters (0-9 and A-Z)
└── README.md              # Project documentation
```

## Installation

No `requirements.txt` is included in this repository. You can run the notebooks in **Google Colab** (where these packages are pre-installed) or in a local Python 3 environment.

### Local Environment Setup

```bash
git clone https://github.com/jatin009v/Captcha-Solver.git
cd Captcha-Solver
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell / CMD):**
  ```powershell
  .venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

Install the required packages based on the notebook imports:

```bash
pip install tensorflow numpy pandas opencv-python scipy jupyter
```

## Setup and Configuration

Both notebooks were originally written for **Google Colab** and mount Google Drive at `/content/drive`. Before executing them, verify or update the file paths to match your environment.

### 1. Inference (`main.ipynb`)

- **Google Colab**:
  - Upload `model5.h5` and your CAPTCHA images to `/content/drive/MyDrive/mosiac/` (or update the paths in the notebook).
- **Local Execution**:
  - Skip or remove the `google.colab` `drive.mount('/content/drive')` cell.
  - Update the model path in the third cell:
    ```python
    model = load_model('model5.h5', compile=True)
    ```
  - Update `image_paths` inside `test()` to point to your local CAPTCHA image file(s).

### 2. Training (`Captcha_Solver.ipynb`)

- The training dataset CSV (`/content/drive/MyDrive/mosiac/filename.csv`) is **not included** in this repository.
- To train the model, provide a headerless CSV file where:
  - Column `0` contains integer class labels (`to_categorical(..., 62)`).
  - Columns `1:` contain `784` pixel values (`28x28` grayscale image), which `img_label()` reshapes to `(rows, 28, 28, 1)`, transposes (`axes=[0, 2, 1, 3]`), and normalizes by dividing by `255`.
- Update the CSV path passed to `img_label(...)` and the checkpoint path in `ModelCheckpoint(...)` (by default `/content/drive/My Drive/modified_weights-{epoch:03d}.h5`).
- If running locally outside Colab, remove the `drive.mount('/content/drive')` and `google.colab.files.download("model5.h5")` cells; `model.save('model5.h5')` writes the trained model to the working directory.

## How to Run

### Running Inference (`main.ipynb`)

1. Open `main.ipynb` in Google Colab or locally via `jupyter notebook main.ipynb`.
2. Verify the paths to `model5.h5` and your input CAPTCHA image(s).
3. Execute the cells in order:
   - Cell 1 mounts Google Drive (Colab only).
   - Cell 2 defines `getMeanArea`, `getRatioArea`, `purify`, and `extract_character`.
   - Cell 3 loads `model5.h5`, defines `predict` and `test`, and invokes `test()`.

### Training the Model (`Captcha_Solver.ipynb`)

1. Open `Captcha_Solver.ipynb` in Google Colab (GPU runtime configured in notebook metadata) or Jupyter Notebook.
2. Provide your labeled CSV dataset and update the input/checkpoint paths.
3. Run all cells to train with `ImageDataGenerator(rotation_range=30, validation_split=0.1)` (`batch_size=300`, `steps_per_epoch=100`, `epochs=40`) and save `model5.h5`.
   - Saved outputs in `Captcha_Solver.ipynb` show Epoch 40 reaching `loss: 0.2232`, `accuracy: 0.9357`, `val_loss: 0.6536`, and `val_accuracy: 0.8100`.

## Usage & Example

Once the helper functions and `model` from `main.ipynb` are loaded, pass a BGR OpenCV image to `predict(image)`:

```python
import cv2

image = cv2.imread("path/to/captcha.jpg")
captcha_decoded = predict(image)
print(captcha_decoded)
```

In the saved cell output of `main.ipynb`, running `test()` on `/content/drive/MyDrive/mosiac/sample captchas/ad0c.jpg` produced:

```text
 HD0C
```

> **Notes on Inference Behavior:**
> - `predict()` initializes `output = ' '`, so returned strings include a leading space.
> - `extract_character()` writes intermediate segmented character images (`character0.jpeg`, `character1.jpeg`, etc.) into the current working directory during execution.

## Notes & Limitations

- **External Data Not Included**: The training CSV (`filename.csv`) and sample image (`sample captchas/ad0c.jpg`) referenced in the notebooks are not present in this repository.
- **Class Mapping**: `character.txt` and `classes` in `main.ipynb` list 36 alphanumeric characters (`0`–`9` and uppercase `A`–`Z`), while `Captcha_Solver.ipynb` configures `num_classes=62` (`Dense(62, activation='softmax')`, with an inline comment noting 62 outputs were kept for class-mapping compatibility).
- **Segmentation Constraint**: `extract_character()` assumes at most 7 segmented characters per CAPTCHA image (triggering recursive retries up to `recursion < 3` when `count > 7`).
