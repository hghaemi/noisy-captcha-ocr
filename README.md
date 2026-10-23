# Noisy CAPTCHA Recognition

A deep-learning pipeline that reads heavily distorted, noisy 5-character CAPTCHAs. It combines a **synthetic CAPTCHA generator** with a **CNN + BiLSTM multi-output classifier** built in TensorFlow/Keras.

> **Results:** 96.5% full-CAPTCHA accuracy and 99.2% character-level accuracy on a held-out set of 200 images.

---

## Table of Contents

- [Motivation](#motivation)
- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Generation](#dataset-generation)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [Limitations](#limitations)

---

## Motivation

The CAPTCHAs this project targets are not clean, textbook examples. They contain dense speckle noise, low-contrast text, and background artifacts, which makes them hard for off-the-shelf OCR.

Started with only **25 real samples**, far too few to train a CNN. Because the original style was hard to reproduce exactly, we designed our own generator that produces thousands of CAPTCHAs that are *at least as noisy* as the originals. The model is trained on this synthetic data.

## Highlights

- **Custom CAPTCHA generator**: random text, colored backgrounds, translucent geometric patterns, rounded borders, bold low-contrast text, and thousands of black noise dots.
- **CNN + BiLSTM architecture**: convolutional layers extract visual features, and bidirectional LSTMs model dependencies along the width of the image.
- **Multi-output head**: one softmax classifier per character position (5 outputs over 36 classes).
- **Memory-friendly training**: a Keras `Sequence` data generator loads images in batches rather than all at once.
- **Training safeguards**: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`, and TensorBoard logging.

## Repository Structure

```
.
├── captcha_generator.py       # Synthetic CAPTCHA dataset generator
├── main.ipynb                 # Preprocessing, model, training, evaluation
├── captcha-presentation.pptx  # Project presentation slides
├── assets/                    # Figures used in this README
├── .gitignore
└── README.md
```

Generated data (`train/`, `captchas/`), trained models (`*.h5`), and logs (`logs/`) are git-ignored. You create them locally by following the steps below.

## Installation

**Requirements:** Python 3.9+ (a GPU is recommended for training).

```bash
git clone https://github.com/hghaemi/noisy-captcha-ocr
cd noisy-captcha-ocr

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install tensorflow numpy opencv-python pillow matplotlib jupyter
```

## Usage

### 1. Generate the training data

```bash
python captcha_generator.py
```

By default this writes 5,000 images to `train/`. Each file is named `{TEXT}_{index}.png` (for example `5QP1T_00000.png`), and the label is read from the part of the filename before the first underscore.

To preview samples without saving a dataset, uncomment `generate_sample_preview(num_samples=10)` in the `__main__` block.

### 2. Prepare a test set

The evaluation code reads images from `captchas/` and expects the same `{TEXT}_{anything}.png` naming. You can either:

- place your own labeled CAPTCHAs in `captchas/`, or
- generate a separate set by setting `OUTPUT_DIR = "captchas"` and `NUM_IMAGES = 200` in `GeneratorConfig`, then running the generator again.

### 3. Train and evaluate

Open `main.ipynb` and run the cells in order:

1. Define configuration, preprocessing, data generator, and model.
2. Run **Training**. The best model is saved to `new_captcha_model.h5`.
3. Run **Test and Evaluation**. This prints accuracy metrics and saves a prediction preview to `predictions_sample.png`.

To run inference on a single image after training:

```python
model = keras.models.load_model(config.MODEL_PATH)
print(predict_captcha(model, "path/to/captcha.png"))
```

## Dataset Generation

`captcha_generator.py` builds each image in layers:

1. **Background**: a random palette from four color schemes (peach, teal, light green, pink).
2. **Patterns**: 3–6 translucent rectangles and 2–4 translucent circles.
3. **Frame**: rounded corners (radius 80) and a thin gray border.
4. **Text**: 5 random characters from `A–Z` and `0–9`, drawn bold and centered with a font size of 80–110.
5. **Noise**: black dots covering about 1.2% of the pixel count, with a radius of 2–4 px.

All parameters (image size, length, character set, palettes, noise density, border style, count) live in `GeneratorConfig` at the top of the file.

## Model Architecture

**Input:** grayscale image, 200 × 700 × 1, normalized to `[0, 1]`.

| Stage | Layers |
|---|---|
| Feature extraction | 4 × [`Conv2D` (3×3, ReLU) → `BatchNorm` → `MaxPool` (2×2) → `Dropout` (0.2)] with 32, 64, 128 and 256 filters |
| Sequence modeling | `Reshape` to (timesteps, features) → `BiLSTM(128)` → `Dropout(0.3)` → `BiLSTM(64)` → `Dropout(0.3)` |
| Output | 5 parallel heads, each `GlobalAveragePooling1D` → `Dense(36, softmax)`, named `char_0` … `char_4` |

Each head classifies one character position, so the five outputs concatenate into the final CAPTCHA string.

## Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate `1e-3` |
| Loss | Sparse categorical cross-entropy, one per character head |
| Batch size | 16 |
| Max epochs | 50 |
| Callbacks | `ModelCheckpoint` (best only), `ReduceLROnPlateau` (factor 0.5, patience 5), `EarlyStopping` (patience 10, restores best weights), `TensorBoard` |
| Hardware | Google Colab, NVIDIA Tesla T4 |

Monitor training live with:

```bash
tensorboard --logdir logs
```

## Results

Evaluated on **200 held-out images**:

| Metric | Score |
|---|---|
| Full-CAPTCHA accuracy (all 5 characters correct) | **96.50%** (193 / 200) |
| Character-level accuracy | **99.20%** |

The training loss is the sum over the five heads. It starts near 18 (about 5 × ln 36, the loss of random guessing) and falls to nearly 0, while mean per-character training accuracy approaches 100%.


`visualize_predictions` in the notebook plots a random mix of correct and incorrect predictions for qualitative error analysis.

## Limitations

- **Synthetic-to-real gap:** the model is trained on generated CAPTCHAs, so accuracy on other styles is not guaranteed without fine-tuning on real samples.
- **Fixed format:** the model expects exactly 5 characters from `A–Z0–9` at 700 × 200 px. Other lengths or alphabets require changing `Config` and retraining.
- **Fonts:** the generator looks for bold Windows fonts (Arial Bold, Arial Black, Impact, Broadway). On Linux or macOS, set `FONT_PATH` in `GeneratorConfig` to a bold `.ttf` file. Otherwise Pillow's small default font is used and the data will look very different.
- **Look-alike characters:** pairs such as `0`/`O` and `1`/`I` are inherently ambiguous in noisy images and are the most likely source of errors.

## Disclaimer

This project is for educational and research purposes. Do not use it to bypass CAPTCHAs on services you do not own or have permission to test.
