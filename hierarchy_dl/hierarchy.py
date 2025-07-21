import os
import json
import time
from os import path
from typing import Optional

import numpy as np
from PIL import Image
from ocrmac import ocrmac
from ultralytics import YOLO

from hierarchy_dl.utils import *

from huggingface_hub import hf_hub_download
cache_dir = "./.models"

ui_elements_model_path = hf_hub_download(
    repo_id="MacPaw/yolov11l-ui-elements-detection",
    filename="ui-elements-detection.pt",
    cache_dir=cache_dir
)

ui_groups_model_path = hf_hub_download(
    repo_id="MacPaw/yolov11l-ui-groups-detection",
    filename="ui-groups-detection.pt",
    cache_dir=cache_dir
)

ui_elements_model = YOLO(ui_elements_model_path)
ui_groups_model = YOLO(ui_groups_model_path)


def generate_hierarchy(
    img: str | Image.Image | np.ndarray,
    save_dir: str = "./results/", 
    save: bool = False, 
    filename: Optional[str] = None,
    flat: bool = False
) -> UIElement:
    """
        Generate UI hierarchy from an image
    """
    # load image
    if isinstance(img, str):
        img_pil = Image.open(img)

    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)

    if isinstance(img, Image.Image):
        img_pil = img

    width, height = img_pil.size

    # detect ui elements
    ui_elements = ui_elements_model(img_pil, verbose=False)[0].boxes
    ui_elements = [UIElement(box, cls) for box, cls in zip(ui_elements.xyxy, ui_elements.cls)]

    # detect ui groups
    ui_groups = ui_groups_model(img_pil, conf=0.5, verbose=False)[0].boxes
    ui_groups = [UIElement(box, "Group") for box in ui_groups.xyxy]

    # ocr
    annotations = ocrmac.OCR(img_pil, language_preference=['en-US']).recognize(px=True)
    annotations = [UIElement(box, "Text", value=val) for val, _, box in annotations]

    # merge texts and elements
    annotations = group_texts(annotations)
    ui_elements = merge_text_and_elements(ui_elements, annotations, iou_threshold=0.2)

    # icons
    ui_elements = caption_buttons(ui_elements, img_pil, batch_size=16)

    if not flat:
        # build tree
        tree = build_tree(ui_groups, ui_elements, (width, height), iou_threshold=0.0)
        clean_tree(tree)

        if len(tree.children) == 1:
            tree = tree.children[0]
    else:
        ui_elements.sort(key=lambda x: x.box[0] ** 2 + x.box[1] ** 2)
        tree = UIElement(
            box=[0, 0, width, height],
            cls="Group",
            value="Screen"
        )
        tree.children = ui_elements

    if save or filename:
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{path.basename(img)}.json" if isinstance(img, str) and not filename else filename
        filename = filename or f"{time.time()}.json"

        full_path = path.join(save_dir, filename)

        with open(full_path, "w", encoding='utf-8') as f:
            json.dump(tree.to_dict(), f, indent=4)

    return tree

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="./screen.png", help="Path to the image")
    parser.add_argument("--save", action="store_true", help="Save the result")
    parser.add_argument("--filename", type=str, default=None, help="Filename to save the result")
    parser.add_argument("--save_dir", type=str, default="./results/", help="Directory to save the result. Default is './results/'")
    parser.add_argument("--flat", action="store_true", help="Generate flat hierarchy (no groups)")
    args = parser.parse_args()

    image = args.image
    save_dir = args.save_dir
    save = args.save
    filename = args.filename
    flat = args.flat

    tree = generate_hierarchy(image, save_dir, save, filename, flat)