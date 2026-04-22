"""
Flask server — runs model inference and serves the interactive web viewer.

Usage:
    python server.py --weights model_best_val_loss_var.pkl

Then open http://localhost:5000 in your browser.
"""

import os
import io
import json
import base64
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn

from flask import Flask, request, jsonify, send_from_directory, abort

from floortrans.models import get_model
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.export_schema import polygons_to_schema, save_schema, load_schema

# ── globals set during startup ────────────────────────────────────────────────
model   = None
device  = None
SPLIT   = [21, 12, 11]
N_CLS   = sum(SPLIT)   # 44

app = Flask(__name__, static_folder="viewer")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(weights_path):
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] device: {device}")

    m = get_model("hg_furukawa_original", 51)
    m.conv4_   = nn.Conv2d(256, N_CLS, bias=True, kernel_size=1)
    m.upsample = nn.ConvTranspose2d(N_CLS, N_CLS, kernel_size=4, stride=4)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(device)
    model = m
    print(f"[server] model loaded from {weights_path}")


def _preprocess(img_bytes):
    """bytes → normalised float tensor (1, 3, H, W)"""
    buf = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    t = torch.tensor(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    t = 2.0 * (t / 255.0) - 1.0          # normalise to [-1, 1]
    return t, height, width, img


def _infer(img_tensor, height, width):
    with torch.no_grad():
        out = model(img_tensor.to(device))
    heatmaps, rooms, icons = split_prediction(out, (height, width), SPLIT)
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), 0.4, [1, 2])
    return polygons, types, room_polygons, room_types


def _img_to_b64(img_rgb):
    """numpy RGB (H,W,3) → base64-encoded PNG string"""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    viewer_path = os.path.join(os.path.dirname(__file__), "viewer", "index.html")
    if not os.path.exists(viewer_path):
        abort(404, "viewer/index.html not found")
    return send_from_directory("viewer", "index.html")


@app.route("/infer", methods=["POST"])
def infer():
    """
    POST /infer
    Body: multipart/form-data with field 'image' (PNG/JPG file)
    Returns: JSON { schema: {...}, image_b64: "..." }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Start server with --weights."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request"}), 400

    img_bytes = request.files["image"].read()
    try:
        img_tensor, height, width, img_rgb = _preprocess(img_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        polygons, types, room_polygons, room_types = _infer(img_tensor, height, width)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    schema = polygons_to_schema(polygons, types, room_polygons, room_types, width, height)
    image_b64 = _img_to_b64(img_rgb)

    return jsonify({"schema": schema, "image_b64": image_b64})


@app.route("/export", methods=["POST"])
def export():
    """
    POST /export
    Body: JSON { schema: {...}, path: "output/result.json" }
    Saves the (corrected) schema to disk and returns { saved: true }.
    """
    body = request.get_json(force=True)
    if not body or "schema" not in body:
        return jsonify({"error": "Missing 'schema' in body"}), 400

    out_path = body.get("path", "output_schema.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_schema(body["schema"], out_path)
    return jsonify({"saved": True, "path": os.path.abspath(out_path)})


@app.route("/load", methods=["POST"])
def load_existing():
    """
    POST /load
    Body: JSON { path: "output/result.json" }
    Loads an existing schema from disk.
    """
    body = request.get_json(force=True)
    if not body or "path" not in body:
        return jsonify({"error": "Missing 'path' in body"}), 400
    try:
        schema = load_schema(body["path"])
        return jsonify({"schema": schema})
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {body['path']}"}), 404


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Floorplan viewer server")
    parser.add_argument("--weights", type=str, default="model_best_val_loss_var.pkl",
                        help="Path to trained model checkpoint (.pkl)")
    parser.add_argument("--host",    type=str, default="127.0.0.1")
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--no-model", action="store_true",
                        help="Start without loading model (viewer-only mode)")
    args = parser.parse_args()

    if not args.no_model:
        if not os.path.exists(args.weights):
            print(f"[server] WARNING: weights file not found at '{args.weights}'. "
                  "Inference endpoint will be unavailable. "
                  "Use --no-model to suppress this warning.")
        else:
            _load_model(args.weights)

    print(f"[server] Starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
