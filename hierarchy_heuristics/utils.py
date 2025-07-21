from collections import defaultdict
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ocrmac import ocrmac
from ultralytics import YOLO
import gdown

from box import *

os.makedirs("models", exist_ok=True)

if not os.path.exists("./models/ui_types_best_v4.pt"):
    print("Downloading element detection model...")
    gdown.download(
        "https://drive.google.com/uc?id=1kFmzkda5k-88Lp59P4LTRSq1HDzyHyQ5",
        "./models/ui_types_best_v4.pt",
        quiet=False,
    )

if not os.path.exists("./models/clickability.pt"):
    print("Downloading clickability model...")
    gdown.download(
        "https://drive.google.com/uc?id=1Q0GPtEgqpFmXOlIVC4Z2wJAzjLToWCLH",
        "./models/clickability.pt",
        quiet=False,
    )

model = YOLO("./models/ui_types_best_v4.pt")
clickability_model = YOLO("./models/clickability.pt")


def plot_app(image, app, curr_depth=0, depths=[]):
    if curr_depth == 0:
        depths.clear()
    img = None
    if curr_depth >= len(depths):
        img = image.copy()
        depths.append(img)
    else:
        img = depths[curr_depth]

    if isinstance(app, Group):
        app = [app]

    for box in app:
        if isinstance(box, Group):
            plot_app(image, box.children, curr_depth=curr_depth + 1)
            cv2.rectangle(img, box.top_left, box.bottom_right, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, box.top_left, box.bottom_right, (255, 0, 0), 2)

        # add class name and rectangle behind it so it is more visible
        cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y1 - 20), (0, 0, 0), -1)
        cv2.putText(
            img,
            Box.id2class[box.cls],
            (box.x1, box.y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    return depths


def plot_all_boxes(image, app, depth = 0):
    cv2.putText(image, str(depth), app.top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    if isinstance(app, Group):
        cv2.rectangle(image, app.top_left, app.bottom_right, (255, 0, 0), 2)

        for child in app.children:
            image = plot_all_boxes(image, child, depth + 1)
        
    else:
        cv2.rectangle(image, app.top_left, app.bottom_right, (0, 0, 255), 2)

    return image


# ## Find texts, bind with boxes and merge
def ocr_image(img, conf_threshold=0.5) -> list[Text]:
    ocr = ocrmac.OCR(Image.fromarray(img), language_preference=["en-US"]).recognize(
        px=True
    )
    return [Text(box, text) for text, conf, box in ocr if conf > conf_threshold]


def bind_text_and_boxes(
    boxes: list[Box], text_boxes: list[Text], iou_threshold=0.5
) -> list[Box]:
    # bind Text to Boxes. Bind Text with largest iou but not less than iou_threshold
    for text_box in text_boxes:
        max_iou = 0
        max_box = None

        for box in boxes:
            iou = text_box.iou(box)

            if iou > max_iou:
                max_iou = iou
                max_box = box

        if max_iou > iou_threshold:
            max_box.text += text_box
        else:
            boxes.append(Box(text_box, Box.class2id["OCRText"], text=text_box))

    return boxes


def mergable(box1: UIBox, box2: UIBox) -> int:
    """
    Returns:
        int: -1 if not mergable, class id if mergable
    """
    if box1.cls == box2.cls:
        return box1.cls

    text_class = [Box.class2id["OCRText"], Box.class2id["AXStaticText"]]

    text_mergeable = [
        Box.class2id["AXButton"],
        Box.class2id["AXLink"],
        Box.class2id["AXStaticText"],
        Box.class2id["AXRadioButton"],
        Box.class2id["AXCheckBox"],
        Box.class2id["AXComboBox"],
        Box.class2id["AXTextField"],
        Box.class2id["AXHeading"],
    ]

    # if static text and button or link
    if box1.cls in text_class and box2.cls in text_mergeable:
        return box2.cls

    if box2.cls in text_class and box1.cls in text_mergeable:
        return box1.cls

    # TODO: forbid. Make image a child of button
    if box1.cls == Box.class2id["AXButton"] and box2.cls == Box.class2id["AXImage"]:
        return Box.class2id["AXButton"]

    if box2.cls == Box.class2id["AXButton"] and box1.cls == Box.class2id["AXImage"]:
        return Box.class2id["AXButton"]

    return -1


def merge_overlapping_boxes(boxes: list[UIBox], iou_threshold=0.5):
    # sort by y1
    boxes.sort(key=lambda box: box.y1)

    i = 0
    while i < len(boxes):
        j = i + 1

        while j < len(boxes):
            if i == j:
                j += 1
                continue

            if boxes[i].x1 < boxes[j].x2 and boxes[j].x1 < boxes[i].x2:  # x-overlap
                merge_cls = mergable(boxes[i], boxes[j])

                if boxes[i].y2 > boxes[j].y1 and merge_cls != -1:  # y-overlap
                    iou = boxes[i].iou(boxes[j])

                    if iou > iou_threshold:
                        boxes[j].merge(boxes[i], inplace=True, cls=merge_cls)

                        del boxes[i]
                        i -= 1

                        if j > i:
                            j -= 1

                        break
            j += 1
            # else:
            #     break

        i += 1

    return boxes


def text_grouping(boxes: list[Box]) -> list[UIBox]:
    """
    https://docs-assets.developer.apple.com/ml-research/papers/screen-recognition-chi-2021.pdf

    We group a TextT1 with a Text below T2 if they satisfy:
    1) they have x-overlap, and
    2) the y-distance between the two texts should be less than a threshold —
     we choose min(T1.heiht, T2.heiht).
    """
    text_boxes = [
        box
        for box in boxes
        if (hasattr(box, "text") and box.text)
        or (isinstance(box, Group) and box.group_type == "text")
    ]

    text_boxes.sort(key=lambda box: box.text.y1)

    for i, box1 in enumerate(text_boxes):
        group = Group(box1.text.box, group_type="text")
        # group = Group(box1.box, group_type="text")
        group.append(box1)

        for j, box2 in enumerate(text_boxes[i + 1 :], start=i + 1):
            if i == j:
                continue

            if box2.parent:
                continue

            bbox = group.children[-1].text

            if (
                bbox.x1 < box2.text.x1 < bbox.x2
                or bbox.x1 < box2.text.x2 < bbox.x2
                or box2.text.x1 < bbox.x1 < box2.text.x2
                or box2.text.x1 < bbox.x2 < box2.text.x2
            ):  # x-overlap
                y_distance = bbox.y_distance(box2.text)
                y_threshold = min(bbox.height, box2.text.height) + 15

                if y_distance < y_threshold:
                    if box2 in boxes:
                        boxes.remove(box2)

                    group.merge_bboxes(box2.text, inplace=True)
                    group.append(box2)

        if len(group.children) > 1:
            if box1 in boxes:
                boxes.remove(box1)

            group.finalize_bbox()
            boxes.append(group)

        else:
            box1.parent = None

    return boxes


# ## Create image groups (image + text)
# Also with button
def get_overlap_percent(image: BBox, text: BBox, coordinate="x") -> float:
    if coordinate == "x":
        overlap = max(0, min(image.x2, text.x2) - max(image.x1, text.x1))
        return overlap / max(image.width, text.width)
    else:
        if text.height > image.height * 1.1:
            return 0

        overlap = max(0, min(image.y2, text.y2) - max(image.y1, text.y1))
        return overlap / max(image.height, text.height)


def merge_images_and_captions(
    boxes: list[UIBox],
    screen_shape: list[int, int],
    x_overlap_percent_threshold=0.25,
    y_distance_threshold=0.02,
    y_overlap_percent_threshold=0.4,
    x_distance_threshold=0.02,
) -> list[UIBox]:
    screen_height, screen_width = screen_shape

    # mergable elements
    elements_with_caption = (
        Box.class2id["AXImage"],
        Box.class2id["AXButton"],
        Box.class2id["AXRadioButton"],
        Box.class2id["AXCheckBox"],
        Box.class2id["AXComboBox"],
    )

    images = [box for box in boxes if box.cls in elements_with_caption]
    text_boxes = [
        box
        for box in boxes
        if (hasattr(box, "text") and box.text)
        or (isinstance(box, Group) and box.group_type == "text")
    ]

    images.sort(key=lambda box: box.y1, reverse=True)
    text_boxes.sort(key=lambda box: box.y1, reverse=True)

    for image in images:
        group = Group(
            image.box, children=[image], group_type=Box.id2class[image.cls][2:]
        )

        for text in text_boxes:
            if (
                text.parent
                and text.parent.cls == Box.class2id["Group"]
                and text.parent.group_type != "text"
            ):
                continue

            if (
                get_overlap_percent(group, text, coordinate="x")
                > x_overlap_percent_threshold
            ):
                y_distance = min(
                    abs(group.y2 - text.y1), abs(group.y2 - text.y2)
                )  # distance between image and text. image must be above text
                y_threshold = y_distance_threshold * screen_height

                if y_distance < y_threshold:
                    if text in boxes:
                        boxes.remove(text)

                    group.merge_bboxes(text, inplace=True)
                    group.append(text)
                    text.parent = group

            if (
                get_overlap_percent(group, text, coordinate="y")
                > y_overlap_percent_threshold
            ):
                x_distance = group.x_distance(
                    text
                )  # image might be on the left or right of text
                x_threshold = x_distance_threshold * screen_width

                if x_distance < x_threshold:
                    if text in boxes:
                        boxes.remove(text)

                    group.merge_bboxes(text, inplace=True)
                    group.append(text)
                    text.parent = group

        if len(group.children) > 1:
            boxes.append(group)
            image.parent = group

            if image in boxes:
                boxes.remove(image)

        else:
            image.parent = None

    return boxes


# ## Group by columns
# elements that are
# 1) close to each other (y coordinate)
# 2) has +- same x1 (maybe x2 too?)
# 3) +- same height?
# 4) has same type?
#
def group_by_column(
    boxes: list[UIBox],
    y_distance_coefficient=1.25,
    width_threshold=40,
    max_width_coefficient=0.55,
) -> list[UIBox]:
    boxes.sort(key=lambda box: box.x1)

    i = 0
    while i < len(boxes):
        box1 = boxes[i]
        group = Group(box1.box, children=[box1], group_type="column")

        j = 0
        while j < len(boxes):
            if i == j:
                j += 1
                continue

            box2 = boxes[j]

            min_height = min(group.height, box2.height)

            if (
                group.y_distance(box2) < y_distance_coefficient * min_height
            ):  # which thresold to use?
                max_width = max(group.width, box2.width)
                min_width = min(group.width, box2.width)

                if (
                    abs(group.x1 - box2.x1) < width_threshold
                    or abs(group.x2 - box2.x2) < width_threshold
                ) and max_width * max_width_coefficient < min_width:
                    if box2 in boxes:
                        j -= 1
                        boxes.remove(box2)

                        if i > j:
                            i -= 1

                    group.merge_bboxes(box2, inplace=True)
                    group.append(box2)

            j += 1

        if len(group.children) > 1:
            if box1 in boxes:
                boxes.remove(box1)
                i -= 1

            box1.parent = group
            boxes.append(group)

        else:
            box1.parent = None

        i += 1

    return boxes


# create histogram of colors in image


def color_histogram(img, threshold):
    # Read the grayscale image
    unique_colors, counts = np.unique(img, return_counts=True)

    # Create a dictionary to hold color areas
    color_area_dict = defaultdict(int)

    # Populate the dictionary with colors and their areas
    for color, count in zip(unique_colors, counts):
        color_area_dict[color] += count

    # Convert dictionary to arrays for faster processing
    colors = np.array(list(color_area_dict.keys()))
    areas = np.array(list(color_area_dict.values()))

    # Merging colors based on the threshold
    merged_colors = []
    merged_areas = []

    while colors.size > 0:
        base_color = colors[0]
        merged_area = areas[0]

        # Create a mask for similar colors
        mask = np.abs(colors - base_color) <= threshold

        # Sum areas of similar colors
        merged_area += areas[mask].sum()

        # Append merged color and area
        merged_colors.append(base_color)
        merged_areas.append(merged_area)

        # Remove merged colors from the list
        colors = colors[~mask]
        areas = areas[~mask]

    # Sort colors by area in descending order
    sorted_indices = np.argsort(merged_areas)[::-1]
    sorted_colors = [merged_colors[i] for i in sorted_indices]
    sorted_areas = [merged_areas[i] for i in sorted_indices]

    return sorted_colors, sorted_areas, sum(sorted_areas)


def get_mask(img, threshold=2, area_threshold=0.01):
    colors_list, sorted_areas, colors_sum = color_histogram(img, threshold)

    for i in range(len(colors_list)):
        color = colors_list[i]

        predicted_perc = sorted_areas[i] / colors_sum
        if predicted_perc < area_threshold:
            break

        lower_bound = np.int16([color]) - threshold
        upper_bound = np.int16([color]) + threshold

        lower_bound = np.where(lower_bound < 0, 0, lower_bound)
        upper_bound = np.where(upper_bound > 255, 255, upper_bound)

        mask = (img >= lower_bound) & (img <= upper_bound)

        elements = np.sum(np.sum(mask))

        percentage = elements / img.size
        if percentage < area_threshold:
            break

        mask = mask.astype(np.uint8) * 255

        yield mask, percentage


def add_color_groups(
    img,
    boxes: list[UIBox],
    verbose=0,
    min_box_threshold=0.03,
    max_box_threshold=0.95,
    color_diff_threshold=2,
    area_threshold=0.01,
):
    min_area = min_box_threshold * img.size
    max_area = max_box_threshold * img.size

    kernel = np.ones((10, 10), np.uint8)

    color_groups: list[Group] = []

    for mask, percentage in get_mask(img, color_diff_threshold, area_threshold):
        if verbose >= 1:
            print(f"Percentage: {percentage * 100:.2f}%")

        original_mask = mask.copy()

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if verbose >= 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            plt.imshow(mask)
            plt.show()
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
            -2
        ]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box_area = w * h

            if not (min_area < box_area < max_area):
                continue

            group = Group((x, y, x + w, y + h), group_type="color")
            color_groups.append(group)

            if verbose >= 1:
                print(f"  - Box area: {box_area / img.size * 100:.5f}%")

            if verbose >= 2:
                mask_copy = original_mask.copy()
                mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(mask_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                plt.imshow(mask_copy)
                plt.show()

    if verbose >= 1:
        print("End")

    color_groups.sort(key=lambda group: group.area)

    # add boxes to color groups
    for color_group in color_groups:
        i = 0
        while i < len(boxes):
            box = boxes[i]

            if box.parent:
                continue

            # box is inside color box
            if color_group.other_is_inside(box, margin=10):
                color_group.append(box)

                if box in boxes:
                    boxes.remove(box)
                    i -= 1

            i += 1

    color_groups = [box for box in color_groups if box.children]

    # merge color groups
    i = 0
    while i < len(color_groups):
        gr1 = color_groups[i]
        j = 0

        while j < len(color_groups):
            if i == j:
                j += 1
                continue

            gr2 = color_groups[j]

            if gr1.other_is_inside(gr2):
                gr1.append(gr2)
                color_groups.remove(gr2)
                j -= 1

                if i > j:
                    i -= 1

            j += 1

        i += 1

    boxes += color_groups

    return boxes


# ## Group by row
def group_by_row(app: Group, y_distance_threshold=20, height_threshold=0.8) -> Group:
    app.children.sort(key=lambda group: group.x1)

    for child in app.children:
        if isinstance(child, Group):
            group_by_row(child, y_distance_threshold, height_threshold)

    i = 0
    while i < len(app.children):
        box1 = app.children[i]

        row_group = Group(box1.box, group_type="row")
        prev_parent = box1.parent
        row_group.append(box1)

        j = 0
        while j < len(app.children):
            box2 = app.children[j]

            if box1 is box2:
                j += 1
                continue

            # if they have +- same y1 or y2 coordinate
            if (
                abs(box2.y1 - row_group.y1) <= y_distance_threshold
                or abs(box2.y2 - row_group.y2) <= y_distance_threshold
            ):
                # if they have +- same height
                min_height = min(row_group.height, box2.height)
                max_height = max(row_group.height, box2.height)

                if min_height / max_height > height_threshold:
                    if box2 in app.children:
                        app.children.remove(box2)
                        j -= 1

                        if i > j:
                            i -= 1

                    row_group.merge_bboxes(box2, inplace=True)
                    row_group.append(box2)

            j += 1

        if len(row_group.children) > 1:
            app.children[i] = row_group
        else:
            box1.parent = prev_parent

        i += 1

    return app


def add_clickability(image, boxes, verbose, conf=0.3):
    clickability_pred = clickability_model(image, verbose=verbose, conf=conf)[0]
    clickability_boxes = zip(clickability_pred.boxes.xyxy, clickability_pred.boxes.cls)
    clickability_boxes = [(BBox(box), cls) for box, cls in clickability_boxes]

    for box in boxes:
        if box.cls in (
            Box.class2id["AXStaticText"],
            Box.class2id["OCRText"],
            Box.class2id["AXImage"],
        ):
            max_iou = 0
            max_elem = None

            for bbox, cls in clickability_boxes:
                curr_iou = box.iou(bbox)

                if curr_iou > 0.1 and curr_iou > max_iou:
                    max_iou = curr_iou
                    max_elem = (bbox, cls)

            if max_elem and max_elem[1] == 1:
                box.cls = Box.class2id["AXButton"]
                box.merge_bboxes(max_elem[0], inplace=True)

    return boxes


def group_elements(img: np.ndarray, **kwargs) -> Group:
    verbose = kwargs.get("verbose", False)
    preds = model(img, conf=kwargs.get("ui_model_conf", 0.3), verbose=verbose)[0]

    img = cv2.cvtColor(preds.orig_img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    boxes, classes = (
        preds.boxes.xyxy,
        preds.boxes.cls,
    )
    boxes = [Box(box, cls) for box, cls in zip(boxes, classes)]

    # use OCR to get more boxes
    ocr_text = ocr_image(img, conf_threshold=kwargs.get("ocr_conf_threshold", 0.3))

    boxes = bind_text_and_boxes(
        boxes,
        ocr_text,
        iou_threshold=kwargs.get("bind_text_and_boxes_iou_threshold", 0.05),
    )

    # # delete boxes with iou > threshold
    boxes = merge_overlapping_boxes(
        boxes, iou_threshold=kwargs.get("merge_overlapping_boxes_iou_threshold", 0.3)
    )

    # add clickability
    boxes = add_clickability(
        img, boxes, verbose, conf=kwargs.get("clickability_model_conf", 0.3)
    )

    # # text grouping
    boxes = text_grouping(boxes)

    boxes = merge_images_and_captions(
        boxes,
        img.shape[:2],
        x_overlap_percent_threshold=kwargs.get(
            "merge_images_and_captions_x_overlap_percent_threshold", 0.25
        ),
        y_distance_threshold=kwargs.get(
            "merge_images_and_captions_y_distance_threshold", 0.02
        ),
        y_overlap_percent_threshold=kwargs.get(
            "merge_images_and_captions_y_overlap_percent_threshold", 0.4
        ),
        x_distance_threshold=kwargs.get(
            "merge_images_and_captions_x_distance_threshold", 0.02
        ),
    )

    boxes = group_by_column(
        boxes,
        y_distance_coefficient=kwargs.get(
            "group_by_column_y_distance_coefficient", 1.25
        ),
        width_threshold=kwargs.get("group_by_column_width_threshold", 40),
        max_width_coefficient=kwargs.get("group_by_column_max_width_coefficient", 0.55),
    )

    # # color groups
    boxes = add_color_groups(
        img_gray,
        boxes,
        verbose=0,
        min_box_threshold=kwargs.get("add_color_groups_min_box_threshold", 0.03),
        max_box_threshold=kwargs.get("add_color_groups_max_box_threshold", 0.95),
        color_diff_threshold=kwargs.get("add_color_groups_color_diff_threshold", 2),
        area_threshold=kwargs.get("add_color_groups_area_threshold", 0.01),
    )

    app = Group((0, 0, img.shape[1], img.shape[0]), children=boxes, group_type="Window")

    app = group_by_row(
        app,
        y_distance_threshold=kwargs.get("group_by_row_y_distance_threshold", 10),
        height_threshold=kwargs.get("group_by_row_height_threshold", 0.8),
    )

    if verbose:
        depths = plot_app(img, app)

        for i, img_plot in enumerate(depths):
            print(f"Depth: {i}")
            plt.figure(figsize=(15, 15))
            plt.imshow(img_plot)
            plt.show()

    return app
