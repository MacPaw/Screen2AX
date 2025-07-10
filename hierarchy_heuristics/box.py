class BBox:
    def __init__(self, box):
        self.box: tuple[int, int, int, int] = tuple(map(int, box))

    def __iter__(self):
        return iter(self.box)

    def __getitem__(self, idx):
        return self.box[idx]

    @property
    def top_left(self):
        return self.box[:2]

    @property
    def bottom_right(self):
        return self.box[2:]

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def area(self):
        return self.width * self.height

    @property
    def x1(self):
        return self.box[0]

    @property
    def y1(self):
        return self.box[1]

    @property
    def x2(self):
        return self.box[2]

    @property
    def y2(self):
        return self.box[3]

    def iou(self, other):
        x1 = max(self.box[0], other.box[0])
        y1 = max(self.box[1], other.box[1])
        x2 = min(self.box[2], other.box[2])
        y2 = min(self.box[3], other.box[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection

        if union == 0:
            return 0

        return intersection / union

    def merge_bboxes(self, other, inplace=False):
        x1 = min(self.box[0], other.box[0])
        y1 = min(self.box[1], other.box[1])
        x2 = max(self.box[2], other.box[2])
        y2 = max(self.box[3], other.box[3])

        if inplace:
            self.box = (x1, y1, x2, y2)
            return self

        return BBox((x1, y1, x2, y2))

    def y_distance(self, other: "BBox"):
        return min(
            abs(self.y1 - other.y2),
            abs(self.y2 - other.y1),
            abs(self.y1 - other.y1),
            abs(self.y2 - other.y2),
        )

    def x_distance(self, other: "BBox"):
        return min(
            abs(self.x1 - other.x2),
            abs(self.x2 - other.x1),
            abs(self.x1 - other.x1),
            abs(self.x2 - other.x2),
        )


class Text(BBox):
    def __init__(self, bbox, text: str) -> None:
        super().__init__(bbox)
        self.text_value = text

    def __repr__(self) -> str:
        return f"Text(box={self.box}, text={self.text_value})"

    def __bool__(self):
        return bool(self.text_value)

    def __add__(self, other):
        if not other:
            return self

        if not self:
            return other

        bbox = (
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )
        text = self.text_value + " " + other.text_value
        return Text(bbox, text)


class UIBox(BBox):
    """Abstract class for UI elements"""

    id2class = {
        0: "AXButton",
        1: "AXCheckBox",
        2: "AXComboBox",
        3: "AXHeading",
        4: "AXImage",
        5: "AXLink",
        6: "AXRadioButton",
        7: "AXScrollBar",
        8: "AXSlider",
        9: "AXStaticText",
        10: "AXTextField",
        11: "TextGroup",
        12: "OCRText",
        13: "Group",
    }

    class2id = {v: k for k, v in id2class.items()}

    def __init__(self, box):
        super().__init__(box)
        self.parent = None
        self.cls = None

    def __repr__(self):
        return f"{self.__class__.__name__}(box={self.box}, cls={self.cls})"

    def merge(self, other, cls=None, inplace=False):
        raise NotImplementedError

    def other_is_inside(self, other, margin=0):
        margin_x1, margin_y1, margin_x2, margin_y2 = (
            margin if isinstance(margin, tuple) else (margin, margin, margin, margin)
        )

        x1 = max(self.x1 + margin_x1, 0)
        y1 = max(self.y1 + margin_y1, 0)
        x2 = self.x2 - margin_x2
        y2 = self.y2 - margin_y2

        return (
            x1 <= other.x1 <= x2
            and y1 <= other.y1 <= y2
            and x1 <= other.x2 <= x2
            and y1 <= other.y2 <= y2
        )


class Group(UIBox):
    group_types = ["window", "row", "text", "column"]

    def __init__(self, box, children=None, group_type="text"):
        super().__init__(box)
        self.cls = self.class2id["Group"]
        self.group_type = group_type
        self.children = children or []

    def __repr__(self):
        return f"Group(box={self.box}, group_type={self.group_type}, children={self.children})"

    def append(self, child):
        self.children.append(child)
        child.parent = self

    def merge(self, other, cls=None, inplace=False):
        if not other:
            return self

        cls = cls or self.cls

        if inplace:
            self.merge_bboxes(other, inplace=True)
            self.cls = cls
            self.children += other.children
            return self

        return Group(self.merge_bboxes(other).box, self.children + other.children)

    def finalize_bbox(self):
        x1, x2, y1, y2 = 9999, 0, 9999, 0

        for child in self.children:
            x1 = min(x1, child.x1)
            y1 = min(y1, child.y1)
            x2 = max(x2, child.x2)
            y2 = max(y2, child.y2)

        self.box = (x1, y1, x2, y2)
        return self


class Box(UIBox):
    def __init__(self, box, cls, text: Text | None = None):
        super().__init__(box)
        self.cls: int = int(cls)
        self.text = text or Text(box, "")

    def __repr__(self):
        return f"Box(box={self.box}, cls={self.cls}, text={self.text})"

    def merge(self, other, cls=None, inplace=False):
        if not other:
            return self

        cls = cls or self.cls

        if inplace:
            self.merge_bboxes(other, inplace=True)
            self.cls = cls
            self.text = self.text + other.text
            return self

        return Box(self.merge_bboxes(other).box, cls, self.text + other.text)
