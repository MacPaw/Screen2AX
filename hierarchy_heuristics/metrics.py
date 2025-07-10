import networkx as nx
import numpy as np


def get_predicted_edges(app, edges, leaf=False):
    if hasattr(app, "children"):
        for child in app.children:
            get_predicted_edges(child, edges)

            if leaf and (not hasattr(child, "children") or not child.children):
                edges.append((app, child))

            if not leaf:
                edges.append((app, child))

    return edges


def get_gt_edges(app, edges, leaf=False):
    if "children" in app:
        for child in app["children"]:
            get_gt_edges(child, edges)

            if leaf and ("children" not in child or not child["children"]):
                edges.append((app, child))
            elif not leaf:
                edges.append((app, child))

    return edges


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (box1[2] - box1[0]) * (box1[3] - box1[1])
        + (box2[2] - box2[0]) * (box2[3] - box2[1])
        - intersection
    )

    return intersection / union


def get_metrics(app, app_json, iou_threshold=0.3, leaf=False):
    preds = []
    get_predicted_edges(app, preds, leaf=leaf)

    gts = []
    get_gt_edges(app_json, gts, leaf=leaf)

    tp = 0
    fp = 0
    fn = 0

    for pred in preds:
        iou_max = 0

        for gt in gts:
            gt_box = gt[0]["xyxy_retina"]

            iou_max = max(iou_max, iou(pred[0].box, gt_box))

        if iou_max >= iou_threshold:
            tp += 1
        else:
            fp += 1

    for gt in gts:
        iou_max = 0
        gt_box = gt[0]["xyxy_retina"]

        for pred in preds:
            iou_max = max(iou_max, iou(pred[0].box, gt_box))

        if iou_max < iou_threshold:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f1


def node_match_function_wrapper(iou_threshold):
    def node_match_function(node1, node2):
        return iou(node1["box"], node2["box"]) >= iou_threshold

    return node_match_function


def calc_ged(app, app_json, iou_threshold=0.3):
    predicted_edges = []
    get_predicted_edges(app, predicted_edges)

    gt_edges = []
    get_gt_edges(app_json, gt_edges)

    G1 = nx.Graph()
    for edge in predicted_edges:
        if edge[0] not in G1.nodes:
            G1.add_node(edge[0], box=edge[0].box)

        if edge[1] not in G1.nodes:
            G1.add_node(edge[1], box=edge[1].box)

        G1.add_edge(edge[0], edge[1])

    G2 = nx.Graph()
    for edge in gt_edges:
        node1 = edge[0].copy()
        node1["children"] = None
        box = node1["xyxy_retina"]
        node1 = tuple(node1.items())

        if node1 not in G2.nodes:
            G2.add_node(node1, box=box)

        node2 = edge[1].copy()
        node2["children"] = None
        box = node2["xyxy_retina"]
        node2 = tuple(node2.items())

        if node2 not in G2.nodes:
            G2.add_node(node2, box=box)

        G2.add_edge(node1, node2)

    return next(
        nx.optimize_graph_edit_distance(
            G1, G2, node_match=node_match_function_wrapper(iou_threshold)
        )
    )


def get_groups(app, groups):
    if hasattr(app, "children"):
        for child in app.children:
            get_groups(child, groups)

        if app.children:
            groups.append(app)

    return groups


def get_gt_groups(app, groups):
    if "children" in app:
        for child in app["children"]:
            get_gt_groups(child, groups)

        if app["children"]:
            groups.append(app)

    return groups


def mean_groups_iou(app, app_json):
    groups = []
    get_groups(app, groups)

    gt_groups = []
    gt_groups = get_gt_groups(app_json, gt_groups)

    ious = []

    for group in groups:
        iou_max = max(
            [iou(group.box, gt_group["xyxy_retina"]) for gt_group in gt_groups]
        )
        ious.append(iou_max)

    # for gt_group in gt_groups:
    #     iou_max = max([iou(group.box, gt_group["xyxy_retina"]) for group in groups])
    #
    #     ious.append(iou_max)

    return np.mean(ious) if ious else 0
