# Floorplan Analysis Pipeline

This repository provides a complete deep learning pipeline for transforming raster floorplan images into structured vector data. It uses a stacked hourglass neural network to perform multi-task semantic segmentation and keypoint detection, followed by a post-processing stage to extract vector graphics.

The original research paper for this work is [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920v1).

## End-to-End Pipeline

The process is broken down into several key stages, managed by a series of Python scripts.

### 1. Data Preparation

Before training, the CubiCasa5K dataset must be downloaded and converted into an efficient format for rapid access.

-   **Download**: The `download_dataset.py` script automates fetching the dataset using the Kaggle API.
-   **Database Creation**: The `create_lmdb.py` script processes the raw image and annotation files into a single, large **LMDB (Lightning Memory-Mapped Database)** file. This binary format stores pre-processed images, segmentation masks, and heatmap coordinates, which dramatically speeds up data loading during training by avoiding repeated parsing of SVG and image files.

    ```bash
    # Download the dataset first
    python download_dataset.py

    # Create LMDB databases for training, validation, and test splits
    python create_lmdb.py --txt train.txt
    python create_lmdb.py --txt val.txt
    python create_lmdb.py --txt test.txt
    ```

### 2. Model Training

The core training logic is handled by `train.py`.

-   **Data Augmentation**: During training, `floortrans/loaders/augmentations.py` applies several on-the-fly transformations to the data to improve model robustness, including random cropping, resizing, rotations, and color jitter.
-   **Training Loop**: The script loads data from the LMDB, feeds it to the model, and computes the loss. It uses the **Adam optimizer** and a learning rate scheduler (`ReduceLROnPlateau`) to adjust the learning rate based on validation performance.
-   **Logging**: Training progress, including losses, variances, and validation metrics, is logged to TensorBoard in the `runs_cubi/` directory.

    ```bash
    # Start training with default parameters
    python train.py

    # Monitor progress
    tensorboard --logdir runs_cubi/
    ```

### 3. Inference and Vectorization

The `inference.py` script provides the full raster-to-vector pipeline, converting a raw floorplan image into a structured set of polygons. An example of this is demonstrated in `samples.ipynb`.

1.  **Image Preprocessing**: The input image is resized so its longest side is 512 pixels, then padded to a 512x512 square and normalized.
2.  **Model Inference**: The pre-trained model predicts a 44-channel output tensor. To improve accuracy, **Test-Time Augmentation (TTA)** is used by default: the image is rotated four times (0°, 90°, 180°, 270°), and the model's predictions are averaged.
3.  **Output Splitting**: The 44-channel tensor is split into three parts:
    *   **Heatmaps (21 channels)**: Representing keypoints like corners and object centers.
    *   **Room Segmentation (12 channels)**: Probabilities for each pixel belonging to a room type.
    *   **Icon Segmentation (11 channels)**: Probabilities for each pixel belonging to an icon type.
4.  **Post-processing & Vectorization**: The `floortrans/post_prosessing.py` script takes the raw model output and converts it into clean vector data. This involves:
    *   Applying a softmax to the room and icon segmentation channels.
    *   Using contour detection and polygon simplification algorithms to extract geometric boundaries for rooms, walls, and icons from the segmentation maps and heatmaps.

## Model Architecture

The core of the pipeline is a **stacked hourglass neural network**, a deep learning architecture originally designed for human pose estimation and adapted here for floorplan feature extraction. This model is implemented in PyTorch in `floortrans/models/hg_furukawa_original.py`.

### Key Architectural Features:

1.  **Stacked Hourglass Modules**: The network is built from multiple "hourglass" modules stacked sequentially. Each module is a self-contained encoder-decoder that refines the feature maps from the previous one. This repeated process of downsampling and upsampling allows the model to learn and integrate features at multiple scales, which is crucial for identifying both large room areas and small icon details.

2.  **Residual Learning**: Within each hourglass module, residual blocks (`Residual`) are used. These blocks employ skip connections that help mitigate the vanishing gradient problem in very deep networks, allowing for more effective training.

3.  **Transfer Learning**: The model leverages transfer learning by initializing its weights from a pre-trained model (`model_1427.pth`). This base model, defined in `floortrans/models/model_1427.py`, is a deep residual network pre-trained on a large-scale computer vision task (likely human pose estimation). This provides a strong initial feature extractor that is then fine-tuned for floorplan analysis.

4.  **Multi-Task Head**: The final layers of the network are replaced with a custom "head" to predict the different components of the floorplan, as described in the inference pipeline.

### Multi-Task Learning

To train these multiple outputs simultaneously, the model uses an **uncertainty-based loss function** (`floortrans.losses.UncertaintyLoss`). This approach, based on the paper ["Multi-Task Learning Using Uncertainty to Weigh Losses"](https://arxiv.org/abs/1705.07115), dynamically balances the different components of the loss. Instead of using static weights, the model learns a "variance" parameter for each task, allowing it to automatically down-weight tasks that are more uncertain or noisy, leading to more stable and effective training.

## Requirements

This repository has been modernized to support **Python 3.10+** and **PyTorch 2.x** on both **CPU** and **CUDA** devices.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Evaluation
Our model weights file can be downloaded [here](https://drive.google.com/file/d/1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK/view?usp=sharing). Once the weights file is in the project folder evaluation can be done. Also you can run the jupyter notebook file to see how the model is performing for different floorplans.
```bash
python eval.py --weights model_best_val_loss_var.pkl
```
Additional option for evaluation can be found in the script file. The results can be found in runs_cubi/ folder. 

## Inference / Backend API

`inference.py` provides a clean, self-contained backend function that takes a raw floorplan image and returns all detected polygons — ready to be consumed by a web application.

### Functions

#### `load_model(weights_path, device=None)`
Loads the Furukawa hourglass checkpoint and returns an eval-mode model on the correct device (CPU or CUDA).

```python
from inference import load_model
model = load_model("model_best_val_loss_var.pkl")
```

#### `predict_floorplan(model, image, *, heatmap_threshold=0.4, use_tta=True, device=None)`
Full end-to-end inference pipeline:

| Step | Detail |
|------|--------|
| **Preprocessing** | Resize longest side to 512 px, zero-pad to 512 × 512, normalise to `[0, 1]` |
| **Inference** | 4-rotation test-time augmentation (0°, 90°, 180°, 270°) — averaged — matching `samples.ipynb` |
| **Postprocessing** | `split_prediction` → softmax rooms/icons → `get_polygons` vector extraction |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Pre-loaded model from `load_model()` |
| `image` | `PIL.Image` or `np.ndarray` | — | Raw floorplan image (any format: RGB, RGBA, greyscale) |
| `heatmap_threshold` | `float` | `0.4` | Minimum heatmap confidence for polygon extraction |
| `use_tta` | `bool` | `True` | Enable 4-rotation test-time augmentation |

**Return value** — `dict`

```python
{
    "polygons":      np.ndarray,           # shape (N, 4, 2)  wall / icon / opening bboxes
    "types":         list[dict],           # {"type": "wall"|"icon", "class": int, ...}
    "room_polygons": list[Polygon],        # shapely Polygon objects per room
    "room_types":    list[dict],           # {"type": "room", "class": int}
}
```

**Class indices**

| Index | Room class | Index | Icon class |
|-------|-----------|-------|-----------|
| 0 | Background | 0 | Empty |
| 1 | Outdoor | 1 | Window |
| 2 | Wall | 2 | Door |
| 3 | Kitchen | 3 | Closet |
| 4 | Living Room | 4 | Electrical Appliance |
| 5 | Bedroom | 5 | Toilet |
| 6 | Bath | 6 | Sink |
| 7 | Hallway | 7 | Sauna Bench |
| 8 | Railing | 8 | Fire Place |
| 9 | Storage | 9 | Bathtub |
| 10 | Garage | 10 | Chimney |
| 11 | Other | | |

### Quick CLI smoke-test

```bash
python inference.py model_best_val_loss_var.pkl path/to/floorplan.png
```

### Web application integration

```python
# At server startup — load once, reuse for every request
from inference import load_model, predict_floorplan
from PIL import Image

model = load_model("model_best_val_loss_var.pkl")

# Inside your request handler (e.g. Flask / FastAPI)
def handle_upload(file_bytes):
    import io
    img = Image.open(io.BytesIO(file_bytes))
    result = predict_floorplan(model, img)

    # Serialise for JSON response
    return {
        "polygons":   result["polygons"].tolist(),
        "types":      result["types"],
        "room_types": result["room_types"],
        # room_polygons are shapely objects — convert as needed
        "rooms": [
            {"class": rt["class"], "coords": list(rp.exterior.coords)}
            for rp, rt in zip(result["room_polygons"], result["room_types"])
        ],
    }
```
