# CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis

Paper: [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920v1)

## Multi-Task Model
The model uses the neural network architecture presented in [Raster-to-Vector: Revisiting Floorplan Transformation](https://github.com/art-programmer/FloorplanTransformation) [1]. The pre- and post-processing parts are modified to suit our dataset, but otherwise the pipeline follows the torch implementation of [1] as much as possible. Our model utilizes the multi-task uncertainty loss function presented in [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115). An example of our trained model's prediction can be found in the samples.ipynb file.

## Dataset
CubiCasa5K is a large-scale floorplan image dataset containing 5000 samples annotated into over 80 floorplan object categories. The dataset annotations are performed in a dense and versatile manner by using polygons for separating the different objects.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/qmarva/cubicasa5k) or [Zenodo](https://zenodo.org/record/2613548).

A helper script is provided to download via Kaggle API:
```bash
python download_dataset.py
```

## Requirements
This repository has been modernized to support **Python 3.10+** and **PyTorch 2.x** on both **CPU** and **CUDA** devices.

Install dependencies:
```bash
pip install -r requirements.txt
```
If you want to use the Dockerfile you need to have docker and [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) installed. We use pre-built image [anibali/pytorch:cuda-9.0](https://github.com/anibali/docker-pytorch) as a starting point and install other required libraries using pip. To create the container run in the:
```bash
docker build -t cubi -f Dockerfile .
```
To start JupyterLab in the container:
```bash
docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --publish 1111:1111 \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD:/app \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  cubi jupyter-lab --port 1111 --ip 0.0.0.0 --no-browser
```
You can now open a terminal in [JupyterLab web interface](http://localhost:1111) to execute more commands in the container.

## Database creation
We create a LMDB database of the dataset, where we store the floorplan image, segmentation tensors and heatmap coordinates. This way we can access the data faster during training and evaluation. The downside however is that the database takes about 105G of hard drive space. There is an option to parse the SVG file on the go but it is slow for training.
Commands to create the database:
```bash
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
python create_lmdb.py --txt train.txt
```

## Train
```bash
python train.py
```
Different training options can be found in the script file. Tensorboard is not included in the docker container. You need to run it outside and point it to cubi_runs/ folder. For each run a new folder is created with a timestamp as the folder name.
```bash
tensorboard --logdir runs_cubi/
```
## Evaluation
Our model weights file can be downloaded [here](https://drive.google.com/file/d/1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK/view?usp=sharing). Once the weights file is in the project folder evaluation can be done. Also you can run the jupyter notebook file to see how the model is performing for different floorplans.
```bash
python eval.py --weights model_best_val_loss_var.pkl
```
Additional option for evaluation can be found in the script file. The results can be found in runs_cubi/ folder. 

## Todo
- Modify create_lmdb.py to save files as uint8 (now using float32 which is the main reason why the lmdb file gets as big as over 100 gbytes).
- Modify augmentations.py to operate with numpy arrays (the reason why it currently utilizes torch tensors is the fact that in our earlier version we applied augmentations to heatmap tensors and not to heatmap dicts which is the correct way to do it)

---

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
