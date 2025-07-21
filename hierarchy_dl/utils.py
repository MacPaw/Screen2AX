from PIL import Image
from ocrmac import ocrmac
from imagehash import average_hash, ImageHash
import cv2
import numpy as np

from .blip import generate_captions


icons_cache = {}


def iou(box1: list[int], box2: list[int]) -> float:
    """
        Calculate IoU between two boxes
    """
    # Extract the coordinates of both boxes
    x1, y1, x2, y2 = box1
    x1_other, y1_other, x2_other, y2_other = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1, x1_other)
    inter_y1 = max(y1, y1_other)
    inter_x2 = min(x2, x2_other)
    inter_y2 = min(y2, y2_other)

    # Check if there is an intersection
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # Area of intersection
    intersection_area = inter_width * inter_height

    # Area of union
    union_area = abs((x2 - x1) * (y2 - y1)) + abs((x2_other - x1_other) * (y2_other - y1_other)) - intersection_area

    if union_area == 0:
        return 0

    # IoU calculation
    return intersection_area / union_area


class UIElement:
    names = [
        'AXButton', 'AXDisclosureTriangle', 'AXImage',
        'AXLink', 'AXTextArea', 'Text',
        'Group'
    ]
    
    def __init__(self, box: list[int], cls: int | str, value: str = None):
        self.value = value or None
        self.box = list(map(int, box)) # [x1, y1, x2, y2]
        self.cls = self.names.index(cls) if isinstance(cls, str) else int(cls)
        self.children: list[UIElement] = []

    
    @property
    def area(self) -> int:
        return abs((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    def iou(self, other: "UIElement") -> float:
        # Extract the coordinates of both boxes
        x1, y1, x2, y2 = self.box
        x1_other, y1_other, x2_other, y2_other = other.box

        # Calculate the coordinates of the intersection rectangle
        inter_x1 = max(x1, x1_other)
        inter_y1 = max(y1, y1_other)
        inter_x2 = min(x2, x2_other)
        inter_y2 = min(y2, y2_other)

        # Check if there is an intersection
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)

        # Area of intersection
        intersection_area = inter_width * inter_height

        # Area of union
        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0

        # IoU calculation
        return intersection_area / union_area


    def merge(self, other: "UIElement"):
        self.box[0] = min(self.box[0], other.box[0])
        self.box[1] = min(self.box[1], other.box[1])
        self.box[2] = max(self.box[2], other.box[2])
        self.box[3] = max(self.box[3], other.box[3])


    def __dict__(self):
        return {
            "cls": self.names[self.cls],
            "value": self.value,
            "box": self.box,
            "children": [child.__dict__() for child in self.children]
        }
    

    def to_dict(self):
        return self.__dict__()
    

    def __repr__(self):
        return str(self.__dict__())


def group_texts(annotations: list[UIElement], max_height_frac: int = 2) -> list[UIElement]:
    """
        Group rows of texts in a single annotation (paragraph)
    """
    # Sort annotations by their vertical position (top Y-coordinate)
    annotations.sort(key=lambda x: x.box[1])

    i = 0
    while i < len(annotations):
        j = i + 1
        ref_height = abs(annotations[i].box[3] - annotations[i].box[1])

        while j < len(annotations):
            box1, box2 = annotations[i].box, annotations[j].box
            height2 = abs(box2[3] - box2[1])

            # Check if annotations are close on the Y-axis
            if abs(box1[3] - box2[1]) > height2 * 0.5:
                j += 1
                continue

            # Check if the height difference is reasonable (similar heights)
            if not (1 / max_height_frac) <= (ref_height / height2) <= max_height_frac: # 1/2 <= h1/h2 <= 2
                j += 1
                continue
            
            # Check if annotations overlap on the X-axis
            if (box1[0] <= box2[0] <= box1[2] or box1[0] <= box2[2] <= box1[2] or
                box2[0] <= box1[0] <= box2[2] or box2[0] <= box1[2] <= box2[2]):
                # Merge the two annotations
                annotations[i].value += f"\n{annotations[j].value}"
                annotations[i].merge(annotations[j])

                # Update height and remove merged annotation
                ref_height = height2
                del annotations[j]
                continue

            j += 1
        i += 1

    return annotations


def merge_text_and_elements(elements: list[UIElement], texts: list[UIElement], iou_threshold=0.2) -> list[UIElement]:
    """
        Merge texts with elements (buttons, images, etc.) based on IoU
    """
    remaining_texts = []

    for text in texts:
        max_iou, best_match = max([(text.iou(element), element) for element in elements], key=lambda x: x[0])


        if max_iou > iou_threshold:
            best_match.value = f"{best_match.value}\n{text.value}" if best_match.value else text.value
            best_match.merge(text)
        else:
            remaining_texts.append(text)

    elements.extend(remaining_texts)
    return elements


def caption_buttons(ui_elements: list[UIElement], image: Image.Image, batch_size: int = 16) -> list[UIElement]:
    """
        Generate captions for buttons
    """
    # find elements that need to be captioned
    to_be_captioned = [e for e in ui_elements if e.cls in (0, 2) and not e.value]

    # calculate crop and hash
    to_be_captioned = [
        ( e, image.crop(e.box), average_hash( image.crop(e.box) ) )
        for e in to_be_captioned
    ]

    # check cache table
    uncached_elements: list[tuple[UIElement, Image.Image, ImageHash]] = []
    for element, cropped, img_hash in to_be_captioned:
        if img_hash in icons_cache:
            element.value = icons_cache[img_hash]
        else:
            uncached_elements.append((element, cropped, img_hash))

    # caption close, minimize, and maximize buttons
    # read in BGR
    system_buttons_img = cv2.imread("./hierarchy_dl/system_buttons.png", cv2.IMREAD_GRAYSCALE)
    cv2_image = np.array(image)[:85, :250] # already in RGB
    cv2_image_bgr = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)

    screenshot_edges = cv2.Canny(system_buttons_img, 50, 200)
    template_edges = cv2.Canny(cv2_image_bgr, 50, 200)

    # find the system buttons using pattern matching. Find area with biggest match
    result = cv2.matchTemplate(screenshot_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    _, maxV, _, max_loc = cv2.minMaxLoc(result)

    if maxV > 0.4:
        x, y = max_loc

        r = 28
        offset = 12

        close_app_location = ([x, y, x + r, y + r], "Close")
        minimize_location = ([x + r + offset, y, x + offset + 2 * r, y + r], "Minimize")
        maximize_location = ([x + 2 * (r + offset), y, x + 2 * (r + offset) + r, y + r], "Zoom")

        uncached_elements.sort(key=lambda x: x[0].box[0] ** 2 + x[0].box[1] ** 2)

        for element, cropped, img_hash in uncached_elements:
            for location, name in [close_app_location, minimize_location, maximize_location]:
                if iou(element.box, location) > 0.1 and element.value is None:
                    element.value = name
                    break

        # remove elements that have been captioned
        uncached_elements = [e for e in uncached_elements if e[0].value is None]

    # try with ocr
    remaining_elements: list[tuple[UIElement, Image.Image, ImageHash]] = []
    for element, cropped, img_hash in uncached_elements:
        vals = ocrmac.OCR(cropped, language_preference=['en-US'], recognition_level="accurate").recognize()

        recognized_texts = [val for val, conf, _ in vals if conf == 1.0]
        
        if recognized_texts:
            element.value = "\n".join(recognized_texts)
            icons_cache[img_hash] = element.value
        else:
            remaining_elements.append((element, cropped, img_hash))

    # generate captions using DeepLearning
    for i in range(0, len(remaining_elements), batch_size):
        batch = remaining_elements[i:i+batch_size]
        crops = [cropped for _, cropped, _ in batch]

        captions = generate_captions(crops)

        for (el, _, img_hash), caption in zip(batch, captions):
            el.value = caption
            icons_cache[img_hash] = caption

    return ui_elements


def build_tree(
    ui_groups: list[UIElement],
    ui_elements: list[UIElement],
    size: tuple[int, int],
    iou_threshold=0.1
) -> UIElement:
    """
        Build a tree from a list of UI elements and groups
    """
    remaining_elements: list[UIElement] = []

    # Assign elements to groups based on IoU
    if ui_groups:
        for element in ui_elements:
            max_iou, best_group = max(
                ((group.iou(element), group) for group in ui_groups), key=lambda x: x[0]
            )

            if max_iou > iou_threshold:
                best_group.children.append(element)
            else:
                remaining_elements.append(element)
    else:
        remaining_elements.extend( ui_elements )

    # Include groups in remaining elements and sort by area (smallest first)
    remaining_elements.extend( ui_groups )
    remaining_elements.sort(key=lambda x: x.area)
    
    merge_occurred = True
    # Continue looping until no merge occurs in a full pass.
    while merge_occurred:
        merge_occurred = False
        i = 0

        # Use a while loop to manage indices when items are removed.
        while i < len(remaining_elements):
            A = remaining_elements[i]

            # Check for a larger element to merge A into.
            for j in range(i + 1, len(remaining_elements)):
                B = remaining_elements[j]

                if A.iou(B) > 0 and B.cls == B.names.index("Group"):
                    # Merge A into B: update B's bounding box and add A as a child.
                    B.merge(A)
                    B.children.append(A)

                    # Remove A from the list.
                    remaining_elements.pop(i)
                    merge_occurred = True

                    # Re-sort remaining_elements as B's area may have increased.
                    remaining_elements.sort(key=lambda x: x.area) # O(n log n), but inserting 1 element could be O(n)

                    # Break out to restart checking from the beginning.
                    break

            else:
                # Only increment if A wasn't merged, because removal shifts indices.
                i += 1

    
    root = UIElement([0, 0, size[0], size[1]], "Group")
    root.children = remaining_elements

    return root


def clean_tree(tree: UIElement):
    """
        Delete empty groups if they are leafs
    """
    i = 0

    while i < len(tree.children):
        child = tree.children[i]

        if child.children:
            clean_tree(child)

        if child.cls == 6:
            if not child.children:
                del tree.children[i]
                continue

            elif len(child.children) == 1:
                tree.children[i] = child.children[0]
                continue

        i += 1

    # Sort children by distance from the parent
    tree.children.sort(key=lambda x: (x.box[0] - tree.box[0]) ** 2 + (x.box[1] - tree.box[1]) ** 2)
