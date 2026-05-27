from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from base.config import ProjectPaths

_DA3_SRC = Path(__file__).resolve().parents[1] / "models" / "Depth-Anything-3" / "src"
if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

from depth_anything_3.api import DepthAnything3

from data.io import write_pfm
from utils.logging_utils import get_logger


def _load_da3(device: torch.device) -> torch.nn.Module:
    paths = ProjectPaths()
    model = DepthAnything3.from_pretrained(str(paths.da3_weights_file))
    return model.to(device).eval()


@torch.no_grad()
def fill_one_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    pred = model(tensor)
    if isinstance(pred, dict) and "depth" in pred:
        depth = pred["depth"]
    else:
        depth = pred
    return depth.squeeze().detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DA3 depth completion for VGGT sparse depth")
    parser.add_argument("--input-dir", required=True, help="dir containing rectified images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ext", default="png")
    args = parser.parse_args()

    logger = get_logger("da3_fill")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_da3(device)
    logger.info("DA3 model loaded")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(in_dir.glob(f"*.{args.ext}"))
    for i, img_path in enumerate(images):
        depth = fill_one_image(model, img_path, device)
        out_path = out_dir / f"{img_path.stem}.pfm"
        write_pfm(str(out_path), depth)
        if (i + 1) % 10 == 0:
            logger.info(f"processed {i + 1}/{len(images)}")
    logger.info("done")


if __name__ == "__main__":
    main()
