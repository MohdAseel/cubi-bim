"""Letterbox the hackathon train images and annotations to 512x512.

Reads from   hackathon_dataset_1/train/{images,annotations}
Writes to    data/hackathon/hackathon/<id>/{F1_scaled.png, F1_original.png, model.svg}
Plus         data/hackathon/hackathon/{train.txt, val.txt}

Per-sample transform pipeline:
    1. SVG coords -> PNG coords:   sx = png_w / svg_w,  sy = png_h / svg_h
    2. PNG coords -> 512x512:      s  = 512 / max(png_h, png_w),   center pad

Combined affine T applied everywhere:
    x_out = svg_x * sx * s + pad_x
    y_out = svg_y * sy * s + pad_y

What gets transformed in the SVG:
    - root <svg> width/height/viewBox -> 512x512
    - <polygon points="..."> children of <g> with id in {Wall,Railing,Window,Door}
      or class containing "Space "  (these are read raw by house.py with NO transform)
    - <g class="FixedFurnitureSet"> transform attr        (T * existing)
    - <g class="FixedFurniture *"> transform attr, but only when its parent is NOT
      a FixedFurnitureSet (because house.py applies M_p only via the Set parent)

Icon-internal coords (BoundaryPolygon polygons, rect x/y/w/h) are intentionally
left untouched -- they're local to the icon and get transformed by M (or M_p*M)
inside house.py at load time.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
from tqdm import tqdm


_MATRIX_RE = re.compile(r"matrix\s*\(([^)]+)\)")


def letterbox_image(img: np.ndarray, target: int = 512):
    """Resize so longest side == target, then center-pad with zeros to target x target."""
    h, w = img.shape[:2]
    s = target / max(h, w)
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return padded, s, pad_left, pad_top


def parse_matrix(s: str):
    m = _MATRIX_RE.search(s)
    if not m:
        return None
    nums = [float(x) for x in m.group(1).replace(",", " ").split() if x]
    if len(nums) != 6:
        return None
    return nums  # [a, b, c, d, e, f]


def fmt_matrix(a, b, c, d, e, f) -> str:
    return f"matrix({a:.6f},{b:.6f},{c:.6f},{d:.6f},{e:.6f},{f:.6f})"


def compose_T_with_matrix(A_global, D_global, E_global, F_global, m):
    """Return T * M as a 6-tuple. T is pure scale+translate (no rotation)."""
    a, b, c, d, e, f = m
    return (
        A_global * a,
        D_global * b,
        A_global * c,
        D_global * d,
        A_global * e + E_global,
        D_global * f + F_global,
    )


def transform_points_attr(points_str: str, A, D, E, F) -> str:
    """Transform every 'x,y' pair in an SVG points string."""
    out = []
    # tokens are whitespace-separated; each token is "x,y"
    for tok in points_str.split():
        tok = tok.strip()
        if not tok:
            continue
        if "," not in tok:
            # malformed token -- keep as-is to avoid silently corrupting
            out.append(tok)
            continue
        parts = tok.split(",")
        try:
            x = float(parts[0])
            y = float(parts[1])
        except (ValueError, IndexError):
            out.append(tok)
            continue
        nx = x * A + E
        ny = y * D + F
        out.append(f"{nx:.4f},{ny:.4f}")
    # NOTE: trailing space is required. svg_utils.PolygonWall.get_points() does
    # split(' ')[:-1] which drops the last token assuming it's empty; without
    # the trailing space it would drop a real point.
    return " ".join(out) + " "


def is_relevant_polygon_owner(elem) -> bool:
    """True if this <g> is one whose direct <polygon> child is read raw by house.py."""
    if elem.nodeType != elem.ELEMENT_NODE or elem.tagName != "g":
        return False
    eid = elem.getAttribute("id") or ""
    cls = elem.getAttribute("class") or ""
    if eid in ("Wall", "Railing", "Window", "Door"):
        return True
    if "Space " in cls or cls.startswith("Space"):
        return True
    return False


def walk(node, fn, *, descend_predicate=None):
    """Pre-order walk. If descend_predicate(node) returns False, skip children."""
    if node.nodeType == node.ELEMENT_NODE:
        fn(node)
        if descend_predicate is not None and not descend_predicate(node):
            return
    for child in list(node.childNodes):
        walk(child, fn, descend_predicate=descend_predicate)


def transform_svg(svg_in_path: str, png_h: int, png_w: int, target: int = 512) -> str:
    """Read SVG, apply combined transform, return XML string."""
    doc = minidom.parse(svg_in_path)
    root = doc.documentElement

    try:
        svg_w = float(root.getAttribute("width"))
        svg_h = float(root.getAttribute("height"))
    except ValueError:
        # fall back to viewBox
        vb = root.getAttribute("viewBox").split()
        svg_w = float(vb[2])
        svg_h = float(vb[3])

    sx = png_w / svg_w
    sy = png_h / svg_h
    s = target / max(png_h, png_w)
    new_w = int(round(png_w * s))
    new_h = int(round(png_h * s))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2

    A = sx * s
    D = sy * s
    E = float(pad_x)
    F = float(pad_y)

    # 1. transform <polygon points> for Wall / Railing / Window / Door / Space groups.
    #    Do NOT descend into FixedFurniture or FixedFurnitureSet -- their inner
    #    polygon points are local to the icon's transform matrix.
    def _is_furniture(node):
        cls = node.getAttribute("class") or ""
        return "FixedFurniture" in cls  # matches both FixedFurnitureSet and FixedFurniture

    def _polygon_visitor(node):
        if not is_relevant_polygon_owner(node):
            return
        for child in node.childNodes:
            if child.nodeType == child.ELEMENT_NODE and child.tagName == "polygon":
                pts = child.getAttribute("points")
                if pts:
                    child.setAttribute("points", transform_points_attr(pts, A, D, E, F))

    def _descend(node):
        # don't recurse into furniture subtrees while looking for room/wall polygons
        if node.nodeType == node.ELEMENT_NODE and _is_furniture(node):
            return False
        return True

    walk(root, _polygon_visitor, descend_predicate=_descend)

    # 2. modify FixedFurnitureSet transforms (one per Set, applied via M_p in house.py)
    handled_sets = set()

    def _set_visitor(node):
        cls = node.getAttribute("class") or ""
        if cls == "FixedFurnitureSet":
            tr = node.getAttribute("transform")
            m = parse_matrix(tr) if tr else None
            if m is None:
                m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            new = compose_T_with_matrix(A, D, E, F, m)
            node.setAttribute("transform", fmt_matrix(*new))
            handled_sets.add(id(node))

    walk(root, _set_visitor)

    # 3. modify FixedFurniture transforms when parent isn't a FixedFurnitureSet
    def _icon_visitor(node):
        cls = node.getAttribute("class") or ""
        if "FixedFurniture " not in cls:
            return
        parent = node.parentNode
        parent_cls = ""
        if parent is not None and hasattr(parent, "getAttribute"):
            parent_cls = parent.getAttribute("class") or ""
        if parent_cls == "FixedFurnitureSet":
            return  # the parent transform already carries T
        tr = node.getAttribute("transform")
        m = parse_matrix(tr) if tr else None
        if m is None:
            m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        new = compose_T_with_matrix(A, D, E, F, m)
        node.setAttribute("transform", fmt_matrix(*new))

    walk(root, _icon_visitor)

    # 4. update root size
    root.setAttribute("width", str(target))
    root.setAttribute("height", str(target))
    root.setAttribute("viewBox", f"0 0 {target} {target}")

    return doc.toxml()


def process_one(image_path: Path, svg_path: Path, out_dir: Path, target: int = 512):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"could not read image {image_path}")
    png_h, png_w = img.shape[:2]
    padded, _, _, _ = letterbox_image(img, target=target)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "F1_scaled.png"), padded)
    cv2.imwrite(str(out_dir / "F1_original.png"), padded)  # same content, pipeline expects both

    svg_xml = transform_svg(str(svg_path), png_h=png_h, png_w=png_w, target=target)
    (out_dir / "model.svg").write_text(svg_xml, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hackathon_dataset_1/train")
    ap.add_argument("--dst", default="data/hackathon/hackathon")
    ap.add_argument("--target", type=int, default=512)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="process only first N pairs (0 = all)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    img_dir = src / "images"
    ann_dir = src / "annotations"

    pairs = []
    for png in sorted(img_dir.glob("*.png")):
        sid = png.stem
        svg = ann_dir / f"{sid}.svg"
        if not svg.exists():
            print(f"[skip] {sid}: no SVG", file=sys.stderr)
            continue
        pairs.append((sid, png, svg))

    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"processing {len(pairs)} pairs -> {dst}")

    failed = []
    succeeded = []
    for sid, png, svg in tqdm(pairs):
        out_dir = dst / sid
        if args.skip_existing and (out_dir / "model.svg").exists():
            succeeded.append(sid)
            continue
        try:
            process_one(png, svg, out_dir, target=args.target)
            succeeded.append(sid)
        except Exception as exc:
            failed.append((sid, str(exc)))
            print(f"[fail] {sid}: {exc}", file=sys.stderr)

    print(f"done: {len(succeeded)} ok, {len(failed)} failed")
    if failed:
        for sid, err in failed[:20]:
            print(f"  {sid}: {err}")

    # write split files (paths are relative to data-path = parent of dst)
    rng = random.Random(args.seed)
    shuffled = succeeded[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.val_ratio))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    def write_split(name, ids):
        path = dst / name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for sid in ids:
                f.write(f"/{sid}/\n")
        print(f"wrote {path} ({len(ids)} entries)")

    write_split("train.txt", train_ids)
    write_split("val.txt", val_ids)


if __name__ == "__main__":
    main()
