import json
import math

import cv2
from bs4 import Tag


def extract_ids_from_output(item):
    """
    Convert the extracted IDs to a list of integers.
    The format is a bit complex to motivate the model to return a more accurate output.
    """
    numbers = []
    if isinstance(item, dict):
        for value in item.values():
            numbers.extend(extract_ids_from_output(value))
    elif isinstance(item, list):
        for element in item:
            numbers.extend(extract_ids_from_output(element))
    elif isinstance(item, int):
        numbers.append(item)
    return numbers


def create_xml_element(element):
    # Only include elements with meaningful content
    if (
        not any(
            [
                element.get("name"),
                element.get("description"),
                element.get("value"),
                element.get("children"),
            ]
        )
        and element["role"] == "AXGroup"
    ):
        return None

    attrib = {
        "role_description": element.get("role_description", ""),
        "id": element["id"],
        "name": element.get("name") if element.get("name") else "",
        "description": (
            element.get("description") if element.get("description") else ""
        ),
        "value": str(element["value"]) if element.get("value") is not None else "",
        "position": element.get("position", "0.0;0.0"),
        "size": element.get("size", "0.0;0.0"),
    }

    xml_element = Tag(name=element["role"], attrs=attrib)
    meaningful_children = list(
        filter(
            lambda x: x is not None,
            [create_xml_element(child) for child in element.get("children", [])],
        )
    )

    # Merge condition: Check if it's a row with only one cell
    if (
        element["role"] == "AXRow"
        and len(meaningful_children) == 1
        and meaningful_children[0].name == "AXCell"
    ):
        cell = meaningful_children[0]
        cell_children = list(cell.children)
        if len(cell_children) == 1:
            merged_element = cell_children[0]
            merged_element["role_description"] = f"merged {attrib['role_description']}"
            return merged_element

    for child in meaningful_children:
        xml_element.append(child)

    # Remove elements that are groups without meaningful children
    if not list(xml_element.children) and element["role"] == "AXGroup":
        return None

    return xml_element


def pretty_print_xml(xml_tree):
    if xml_tree is None:
        return ""

    if type(xml_tree) == str:
        return xml_tree

    return xml_tree.prettify()


def map_ids(element, id_mapping, current_id=1):
    original_id = element["id"]
    id_mapping[current_id] = original_id
    element["id"] = str(current_id)
    current_id += 1

    for child in element.find_all(recursive=False):
        current_id = map_ids(child, id_mapping, current_id)

    return current_id


def json_to_xml(json_obj):
    id_mapping = {}
    root_element = create_xml_element(json_obj)
    if root_element is None:
        return None, id_mapping

    # Map ids of elements
    map_ids(root_element, id_mapping)
    return root_element, id_mapping


def add_ids_to_json(json_obj, curr_id=0):
    json_obj["id"] = curr_id
    curr_id += 1

    for child in json_obj["children"]:
        _, curr_id = add_ids_to_json(child, curr_id)
        curr_id += 1

    return json_obj, curr_id


def format_json(json_object):
    new_json_object = {}
    new_json_object["@children"] = []

    for key, value in json_object.items():
        print(key, value)
        if not key.startswith("@"):
            if isinstance(value, dict):
                format_json(value)
                value["@role"] = key
                new_json_object["@children"].append(value)
            else:
                for child in value:
                    format_json(child)
                    child["@role"] = key
                    new_json_object["@children"].append(value)
        else:
            new_json_object[key] = value

    return new_json_object


def xml2dict(xml_object: Tag, width, height):
    dict_object = {"role": xml_object.name, **xml_object.attrs, "children": list()}
    position = xml_object.get("position", "0.0;0.0")
    size = xml_object.get("size", "0.0;0.0")

    position = list(map(float, position.split(";"))) if position else (0, 0)
    size = list(map(float, size.split(";"))) if size else (0, 0)

    try:
        position = list(map(int, position))
    except Exception as e:
        if abs(position[0]) == float("inf"):
            position[0] = 0 if position[0] < 0 else width // 2
        elif math.isnan(position[0]):
            position[0] = 0
        else:
            position[0] = int(position[0])

        if abs(position[1]) == float("inf"):
            position[1] = 0 if position[1] < 0 else height // 2
        elif math.isnan(position[1]):
            position[1] = 0
        else:
            position[1] = int(position[1])

    try:
        size = list(map(int, size))
    except Exception as e:
        if abs(size[0]) == float("inf"):
            size[0] = 0 if position[0] < 0 else width // 2 - position[0]
        elif math.isnan(size[0]):
            size[0] = 0
        else:
            size[0] = int(size[0])

        if abs(size[1]) == float("inf"):
            size[1] = 0 if size[1] < 0 else height // 2 - position[1]
        elif math.isnan(size[1]):
            size[1] = 0
        else:
            size[1] = int(size[1])

    del dict_object["position"]
    del dict_object["size"]

    dict_object["xyxy"] = [*position, position[0] + size[0], position[1] + size[1]]
    dict_object["xyxy_retina"] = [
        2 * position[0],
        2 * position[1],
        2 * (position[0] + size[0]),
        2 * (position[1] + size[1]),
    ]

    if dict_object["role"] == "AXWindow":
        dict_object["xyxy"] = [0, 0, width // 2, height // 2]
        dict_object["xyxy_retina"] = [0, 0, width, height]

    for child in xml_object.children:
        dict_object["children"].append(xml2dict(child, width, height))

    return dict_object


def simplify_tree(json_path: str, width, height):
    with open(json_path, "r") as f:
        json_object = json.load(f)

    json_object, _ = add_ids_to_json(json_object)

    root_element, id_mapping = json_to_xml(json_object)

    # convert to dict
    root_element_json = xml2dict(root_element, width, height)

    # save
    with open(f"{json_path[:-5]}_simplified.json", "w") as f:
        json.dump(root_element_json, f, indent=2)


types = [
    "AXComboBox",
    "AXLink",
    "AXMenuBar",
    "AXPage",
    "AXHeading",
    "AXListMarker",
    "AXList",
    "AXOpaqueProviderGroup",
    "AXDateTimeArea",
    "AXSlider",
    "AXWindow",
    "AXDisclosureTriangle",
    "AXSheet",
    "AXMenu",
    "AXMenuButton",
    "No role",
    "AXCell",
    "AXColorWell",
    "AXTextField",
    "AXIncrementor",
    "AXScrollArea",
    "AXButton",
    "AXPopover",
    "AXColumn",
    "JavaAxIgnore",
    "AXRadioButton",
    "AXLevelIndicator",
    "AXMenuItem",
    "AXStaticText",
    "AXRadioGroup",
    "AXGroup",
    "AXScrollBar",
    "AXSplitGroup",
    "AXToolbar",
    "AXRuler",
    "AXProgressIndicator",
    "AXValueIndicator",
    "AXTabGroup",
    "AXGrowArea",
    "AXImage",
    "AXRow",
    "AXGenericElement",
    "AXWebArea",
    "AXCheckBox",
    "AXOutline",
    "AXGrid",
    "AXBrowser",
    "AXSplitter",
    "AXBusyIndicator",
    "AXUnknown",
    "AXTextArea",
    "AXPopUpButton",
    "AXTable",
]

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


def plot_json(img, json_object, save_path=None):
    color = bgr_colors[types.index(json_object["role"]) % len(bgr_colors)]

    x1, y1, x2, y2 = json_object["xyxy_retina"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = json_object["role"]
    text_color = (255, 255, 255)

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    # Prints the text
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    for child in json_object["children"]:
        plot_json(img, child)

    if save_path is not None:
        cv2.imwrite(save_path, img)
