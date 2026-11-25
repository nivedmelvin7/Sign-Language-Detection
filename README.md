# Live ASL fingerspelling recognition (PyTorch + OpenCV + MediaPipe)

`sign_language_live.ipynb` trains a CNN on the Kaggle **Sign Language MNIST** dataset on your GPU, then recognises static ASL letters (A to Y, without J and Z) live from your laptop webcam. Every image processing step is written out in the notebook and explained in markdown.

## 1. Setup

You need Python 3.14, a webcam and, ideally, an NVIDIA GPU with a recent driver.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install the CUDA build of PyTorch first

Go to <https://pytorch.org/get-started/locally/>, choose your OS, **Pip**, **Python** and a **CUDA** version, and run the command it shows. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

This matters most on **Windows**, where a plain `pip install torch` installs a CPU-only build. On Linux the default PyPI wheels already include CUDA. On an Apple Silicon Mac the default wheels use the GPU through MPS.

Then install everything else:

```bash
pip install -r requirements.txt
```

### Get the data

The dataset is not stored in this repository. Either run `main.ipynb`, which downloads *Sign Language MNIST* (datamunge) from Kaggle into `./dataset/` with `kagglehub`, or download it by hand and put `sign_mnist_train.csv` and `sign_mnist_test.csv` in a folder. The notebook expects `./dataset/` by default. If you use another folder, change `DATA_DIR` at the top of section 2.

## 2. Confirm the GPU is detected

Quick check from a terminal:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

You want a CUDA version (not `None`) and `True`. In the notebook, section 3 ("GPU check") prints the PyTorch build, the GPU name, VRAM and cuDNN version, then runs a small matrix multiplication on the GPU.

With `CONFIG["REQUIRE_GPU"] = True` (the default), the notebook stops with an explanation if no GPU is found:

- `torch.version.cuda` is `None`: you have a CPU-only PyTorch build. Reinstall with the pytorch.org command above, then restart the Jupyter kernel.
- `torch.version.cuda` has a version but no GPU is visible: check the NVIDIA driver with `nvidia-smi`.

To run on the CPU anyway (training is slower but still only a few minutes), set `CONFIG["REQUIRE_GPU"] = False`.

## 3. Run the notebook

```bash
jupyter lab sign_language_live.ipynb      # or open it in VS Code
```

Set `DATA_DIR` and `CAM_INDEX` in section 2, then **Run All**. On an RTX 3060 laptop GPU the whole notebook up to the webcam section takes about a minute and a half (training itself takes under a minute). It saves the model to `artifacts/`. The first run also downloads MediaPipe's `hand_landmarker.task` (about 7.5 MB).

## 4. The live demo

Section 16 opens an OpenCV window (not inline in the notebook). Click the window so it has keyboard focus.

| Key | Action |
|---|---|
| `q` | quit (the camera is always released, even after an error) |
| `m` | toggle mirroring of the model input |
| `h` | toggle CLAHE contrast enhancement (useful in poor light) |
| `space` | add a space to the typed text |
| `backspace` | delete the last character |
| `c` | clear the text |

Hold a letter steadily for about one second to type it. The enlarged image in the top-right corner is exactly what the CNN sees. Compare it with the dataset samples in section 4 when predictions are wrong.

**Running only the live demo after a restart** (no retraining): run the code cells of sections 2 and 3, the first code cell of section 5, the first code cell of section 9, the second code cell of section 15 (`load_model`), then section 16.

**Tips:** use a plain, light background; light your hand evenly from the front; keep the hand upright, still and filling the green box. If the camera does not open, close other apps using it (Zoom, Teams, browser tabs) and try `CAM_INDEX = 1`. On Linux under Wayland, if the OpenCV window does not appear, start Jupyter with `QT_QPA_PLATFORM=xcb`.

## 5. Optional: fine-tune on your own hand

1. Set `CONFIG["RUN_CAPTURE"] = True` and run the capture cell in section 17. For each letter press `space` to record (it saves 150 crops to `my_data/<letter>/`), `n` for the next letter and `q` to quit.
2. Run the fine-tuning cell. It fine-tunes on a mix of your images and the original data at a low learning rate, and reports accuracy on a held-out part of your images before and after. The fine-tuned weights go to `artifacts/sign_cnn_finetuned.pt`, and the next cell starts the live demo with them.

## Files

| File | Contents |
|---|---|
| `sign_language_live.ipynb` | the notebook |
| `requirements.txt` | Python dependencies |
| `artifacts/sign_cnn_best.pt` | trained weights (state_dict) |
| `artifacts/label_map.json` | class index -> letter |
| `artifacts/config.json` | the full `CONFIG` used for training |
| `hand_landmarker.task` | MediaPipe hand model (downloaded automatically) |
| `my_data/` | your own captured images (optional) |
