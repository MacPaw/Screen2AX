from typing import List, TypedDict, Dict, Optional

class UIElement(TypedDict):
    box: List[int]
    cls: str
    value: Optional[str]
    parent: "UIElement"
    children: List["UIElement"]
    prev: "UIElement"
    next: "UIElement"
    index: int

def ordinal(n: int) -> str:
    """
    Returns the ordinal representation of an integer (e.g., 1 -> '1st').
    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]

    return f"{n}{suffix}"


def iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (List[int]): The first bounding box.
        box2 (List[int]): The second bounding box.

    Returns:
        float: The IoU value.
    """
    x1, y1, x2, y2 = box1
    x1_other, y1_other, x2_other, y2_other = box2

    inter_x1 = max(x1, x1_other)
    inter_y1 = max(y1, y1_other)
    inter_x2 = min(x2, x2_other)
    inter_y2 = min(y2, y2_other)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_other - x1_other) * (y2_other - y1_other)
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


def remove_small_groups(node: Dict, n: int):
    if "children" not in node:
        return

    # First process children recursively.
    for child in node["children"]:
        remove_small_groups(child, n)

    # Now, scan through the children list and promote any group that has fewer than n children.
    i = 0
    while i < len(node["children"]): # skip root node
        child = node["children"][i]

        if "children" in child and len(child["children"]) < n and child["cls"].lower().endswith("group"):
            # Promote the child’s children into the parent's children list.
            promoted = child["children"]
            # Update each promoted child's parent pointer.
            for p in promoted:
                p["parent"] = node

            child["parent"] = None
            child["children"] = []
            child["to_remove"] = True

            # Replace the small group with its children.
            node["children"] = node["children"][:i] + promoted + node["children"][i+1:]
            # After inserting the promoted children, recheck the new entries.
        else:
            i += 1

    def sort_heuristic(c: Dict) -> float:
        """
        Heuristic function to sort children based on their bounding box coordinates.
        """
        x, y = c["box"][0], c["box"][1]

        if c["parent"] is None:
            return y ** 2 + 0.5 * x ** 2
        
        # If the child has a parent, we want to subtract the parent's coordinates
        parent_x, parent_y = c["parent"]["box"][0], c["parent"]["box"][1]
        return (y - parent_y) + 0.3 * (x - parent_x)

    # re-sort the children list
    node["children"].sort(key=sort_heuristic)

    # Remove any children marked for removal.
    node["children"] = [c for c in node["children"] if "to_remove" not in c]

    # Update sibling links for the (possibly updated) children list.
    for j, c in enumerate(node["children"]):
        c["index"] = j + 1
        c["prev"] = node["children"][j - 1] if j > 0 else node["children"][-1]
        c["next"] = node["children"][j + 1] if j < len(node["children"]) - 1 else node["children"][0]


def add_description_to_groups(node: Dict, max_len: int = 100) -> None:
    """
    Adds a description to group nodes based on their children.

    Args:
        node (Dict): The node to process.
    """
    if "children" not in node:
        return
    
    for child in node["children"]:
        add_description_to_groups(child)

    if node["cls"].lower().endswith("group"):
        descriptions = []

        for child in node["children"]:
            if child["value"] is not None:
                descriptions.append(child["value"])

        value = descriptions[0] if descriptions else ""
        for item in descriptions[1:]:
            new_value = f"{value}, {item}" if value else item

            if len(new_value) > max_len:
                new_value = new_value[:max_len] + ", and other..."
                break

            value = new_value


        if len(value) > max_len:
            value = value[:max_len] + ", and other..."

        node["value"] = value


def update_accessibility_data(data: Dict, n: int = 0) -> UIElement:
    """
    Updates the accessibility data by adding parent, next, and previous links to nodes.
    Also, remove groups that have less than n children.

    Args:
        data (Dict): The original accessibility data.
        n (int): The minimum number of children a group must have to be retained.
            Defaults to 0.

    Returns:
        Dict: The updated accessibility data.
    """
    stack = [data]
    data["parent"] = None
    data["next"] = None
    data["prev"] = None
    data["index"] = 1

    leaves = []

    while stack:
        node = stack.pop()

        # Scale down the bounding box coordinates.
        node["box"] = [coord // 2 for coord in node["box"]]

        if "children" not in node:
            continue

        if len(node["children"]) == 0:
            leaves.append(node)
            continue

        node["children"].sort(key=lambda x: (x["box"][1] ** 2 + 0.5 * x["box"][0] ** 2))

        for i, child in enumerate(node["children"]):
            child["parent"] = node
            child["index"] = i + 1
            child["prev"] = node["children"][i - 1] if i > 0 else node["children"][-1]
            child["next"] = (
                node["children"][i + 1]
                if i < len(node["children"]) - 1
                else node["children"][0]
            )

            if child["value"] and "|" in child["value"]:
                child["value"] = child["value"].split("|")[0].strip()

            stack.append(child)

    # Remove groups with less than n children
    remove_small_groups(data, n)

    add_description_to_groups(data, max_len=30)

    return data