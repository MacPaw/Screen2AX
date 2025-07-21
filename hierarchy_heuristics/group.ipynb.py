## Imports
import json
import os
import signal
from glob import glob

import cv2
from tqdm import tqdm

from box import *
from metrics import *
from utils import group_elements

config = {
    # YOLO model
    "ui_model_conf": 0.3,  # [0, 1]
    # OCR model
    "ocr_conf_threshold": 0.3,  # [0, 1]
    # bind_text_and_boxes
    "bind_text_and_boxes_iou_threshold": 0.05,  # [0, 1]
    # merge_overlapping_boxes
    "merge_overlapping_boxes_iou_threshold": 0.3,  # [0, 1]
    # clickability confidence
    "clickability_model_conf": 0.3,  # [0, 1]
    # merge_images_and_captions
    "merge_images_and_captions_x_overlap_percent_threshold": 0.25,  # [0, 1]
    "merge_images_and_captions_y_distance_threshold": 0.02,  # [0, 1]
    "merge_images_and_captions_y_overlap_percent_threshold": 0.4,  # [0, 1]
    "merge_images_and_captions_x_distance_threshold": 0.02,  # [0, 1]
    # group_by_column
    "group_by_column_y_distance_coefficient": 1.25,  # [0, inf)
    "group_by_column_width_threshold": 40,  # [0, inf)
    "group_by_column_max_width_coefficient": 0.55,  # [0, 1]
    # add_color_groups
    "add_color_groups_min_box_threshold": 0.03,  # [0, 1]
    "add_color_groups_max_box_threshold": 0.95,  # [0, 1]
    "add_color_groups_color_diff_threshold": 2,  # [0, 255]
    "add_color_groups_area_threshold": 0.01,  # [0, 1]
    # group_by_row
    "group_by_row_y_distance_threshold": 50,  # [0, inf)
    "group_by_row_height_threshold": 0.8,  # [0, 1]
}

if __name__ == "__main__":
    [os.remove(img) for img in glob("./test/ideal-test/*/*/*.png") if "predicted" in img]


    images = glob("./test/ideal-test/*/*/*.png")
    images = list(filter(lambda x: "segmented" not in x, images))
    images.sort()

    jsons = list(sorted(glob("./test/ideal-test/*/*/*_simplified.json")))

    dataset = list(zip(images, jsons))


    def handler(signum, frame):
        raise Exception("Timeout")


    apps = [group_elements(img, **config) for img, _ in tqdm(dataset)]


    apps_json = [json.load(open(json_path)) for _, json_path in tqdm(dataset)]

    bgr_colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (192, 192, 192),  # Silver
        (128, 128, 128),  # Gray
        (255, 165, 0),  # Orange
        (255, 192, 203),  # Pink
        (210, 105, 30),  # Chocolate
        (34, 139, 34),  # Forest Green
        (255, 215, 0),  # Gold
        (135, 206, 250),  # Sky Blue
    ]


    def plot_app(img, app, depth=0):
        color_id = app.cls
        label = Box.id2class[app.cls]

        if isinstance(app, Group):
            label += f"[{app.group_type} {depth}]"
            color_id += (
                Group.group_types.index(app.group_type)
                if app.group_type in Group.group_types
                else len(Group.group_types)
            )

            color = bgr_colors[color_id % len(bgr_colors)]

            x1, y1, x2, y2 = app.box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            text_color = (255, 255, 255)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Prints the text
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(
                img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1
            )

        if hasattr(app, "children"):
            for child in app.children:
                plot_app(img, child, depth + 1)

        # if save_path is not None:
        #     cv2.imwrite(save_path, img)
        return img


    for app, (img, _) in zip(apps, dataset):
        image = cv2.imread(img)
        image = plot_app(image, app)

        segmented_image_path = f"{img[:-4]}_simplified-segmented.png"
        original_segmented_image = cv2.imread(segmented_image_path)

        output_image = np.concatenate((original_segmented_image, image), axis=1)
        img_name = f"{img.split('/')[-1]}"
        cv2.imwrite(f"./test-output/{img_name}", output_image)

    metrics = []
    leaf_metrics = []
    geds = []
    mIoU = []

    for app, app_json in tqdm(zip(apps, apps_json), total=len(apps)):
        ## F1 for edges
        metrics.append(get_metrics(app, app_json, iou_threshold=0.3, leaf=False))
        leaf_metrics.append(get_metrics(app, app_json, iou_threshold=0.3, leaf=True))

        ## GED
        # try to get GED for 10 seconds. In other case add length of Ground Truth
        try:
            signal.signal(signal.SIGALRM, handler)

            signal.alarm(10)  # 10 seconds
            ged = calc_ged(app, app_json, iou_threshold=0.5)
            signal.alarm(0)

            geds.append(ged)

        except Exception as e:
            gt_edges = []
            get_gt_edges(app_json, gt_edges)
            length = len(gt_edges)
            geds.append(length)

        ## Average IoU for groups
        mIoU.append(mean_groups_iou(app, app_json))

    # ## Results
    print(
        f"""
    Mean precision {np.mean([i[0] for i in metrics]):.2f} ± {np.std([i[0] for i in metrics]):.2f}
    Mean recall {np.mean([i[1] for i in metrics]):.2f} ± {np.std([i[1] for i in metrics]):.2f}
    Mean F1 score {np.mean([i[2] for i in metrics]):.2f} ± {np.std([i[2] for i in metrics]):.2f}
    """
    )

    print(
        f"""
    Mean leaf precision {np.mean([i[0] for i in leaf_metrics]):.2f} ± {np.std([i[0] for i in leaf_metrics]):.2f}
    Mean leaf recall {np.mean([i[1] for i in leaf_metrics]):.2f} ± {np.std([i[1] for i in leaf_metrics]):.2f}
    Mean leaf F1 score {np.mean([i[2] for i in leaf_metrics]):.2f} ± {np.std([i[2] for i in leaf_metrics]):.2f}
    """
    )

    print(f"Mean GEDs {np.mean(geds):.2f} ± {np.std(geds):.2f}")

    print(f"Mean average IoU for groups {np.mean(mIoU):.2f} ± {np.std(mIoU):.2f}")
