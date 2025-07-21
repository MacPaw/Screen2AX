import cv2
from utils import group_elements, plot_app, plot_all_boxes
import time

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
    img = cv2.imread("./visual-test-images/test4.png")

    start = time.time()
    app = group_elements(img, verbose=False, **config)
    print("Time:", time.time() - start)

    img = plot_all_boxes(img, app)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()