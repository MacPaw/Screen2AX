import os
import time
import argparse
import subprocess
import tkinter as tk
from typing import Tuple, Dict

from AppKit import NSApp
from PIL import Image, ImageChops
from pynput.mouse import Button, Controller
from macapptree import get_tree, get_app_bundle

from hierarchy_dl.hierarchy import generate_hierarchy

from screen_reader.utils import *
from screen_reader.screenshot import (
    get_app_info,
    open_app_in_foreground,
    screenshot_app,
    get_app_name,
)


# Global mouse controller
mouse = Controller()

# Global variables and constants
BUNDLE_ID = None
SAY_RATE = 190
WELCOME = True
HELP = True
SYSTEM_ACCESSIBILITY = False
VOICE = "Daniel"
SKIP_GROUPS_SIZE = 5

WELCOME_MESSAGE = "Welcome to the ScreenReader."
HELP_MESSAGE = (
    "Use the arrow keys to navigate the UI elements. [[slnc 500]]\n"
    "- Press the down arrow to navigate inside a group.\n"
    "- Press the up arrow to navigate to the parent group.\n"
    "- Press the left and right arrows to navigate to the previous and next elements.\n"
    "- Press the space bar to click on a button or link.\n"
    "- Press h to hear the instructions again.\n"
    "- Press q to quit the ScreenReader."
)
SCREEN = None
SAY_PROCESS: subprocess.Popen = subprocess.Popen(["say", ""])
ACTIVE_ELEMENT: UIElement = {}


def make_click_through(window: tk.Tk) -> None:
    """
    Enables click-through for the given Tkinter window.
    """
    ns_window = NSApp().windows()[0]
    ns_window.setIgnoresMouseEvents_(True)


def wake_up(root: tk.Tk) -> None:
    """
    Brings the Tkinter window to the foreground.
    """
    root.deiconify()
    root.lift()
    root.focus_force()
    NSApp.activateIgnoringOtherApps_(True)
    root.update()


def create_app(position: Tuple[int, int, int, int]) -> Tuple[tk.Tk, tk.Canvas]:
    """
    Creates and returns a transparent Tkinter window along with its drawing canvas.

    Args:
        position (Tuple[int, int, int, int]): (x, y, width, height).

    Returns:
        Tuple[tk.Tk, tk.Canvas]: The Tkinter root window and its associated canvas.
    """
    root = tk.Tk()
    root.title("ScreenReader")
    root.overrideredirect(True)
    root.wm_attributes("-transparent", True)
    root.config(bg="systemTransparent")

    x, y, width, height = map(int, position)
    root.geometry(f"{width}x{height}+{x}+{y}")

    canvas = tk.Canvas(root, width=width, height=height, bg="systemTransparent")
    canvas.pack()

    # Allow the window to be click-through after a short delay.
    root.after(100, lambda: make_click_through(root))
    return root, canvas


def get_message(node: UIElement) -> str:
    """
    Generates a message based on the properties of a node in the accessibility tree.

    Args:
        node (UIElement): A node from the accessibility data.

    Returns:
        str: The generated message.
    """
    message = ""
    place = ""

    node_value = node.get("value", "")
    node_cls = node.get("cls", "")

    if node_cls.startswith("AX"):
        node_cls = node_cls[2:]

    if node_cls.endswith("Group"):
        children = node.get("children", [])
        message = f"{node_cls} with {len(children)} items. {'Groups content: ' if node_value else ''} {node_value}. {place}. [[slnc 500]] You are currently in a group. To interact with the items, press the down arrow."
    elif node_cls.endswith("Button"):
        message = f"{node_cls} with description {node_value}. You are currently on a button. To click, press the space bar."
    elif node_cls.endswith("Image"):
        message = f"{node_cls} with description {node_value}."
    elif node_cls.endswith("Link"):
        message = f"{node_cls} {node_value}. You are currently on a link. To click, press the space bar."
    elif node_cls.endswith("Text"):
        message = f"{node_cls} detected: {node_value}. "
    else:
        children = node.get("children", [])
        message = f"{node_cls}, {node_value}. {f'[[slnc 500]] {node_cls} with {len(children)} items.' if children else ''}"

    parent = node.get("parent")
    if parent and len(parent.get("children", [])) > 1:
        object_type = "group" if node["cls"].endswith("Group") else "element"
        place = f"{ordinal(node['index'])} {object_type} out of {len(node['parent']['children'])}"
        place = f"[[slnc 500]] {place}"

        if message and not node_cls.endswith("Group"):
            message += f" {place}."

    return message


def read_message(message: str, wait: bool = False) -> None:
    """
    Uses the macOS 'say' command to read a message aloud.

    Args:
        message (str): The message to be read.
        wait (bool): If True, waits for the current speech to finish before proceeding.
    """
    global SAY_PROCESS

    if not message:
        return

    if wait:
        SAY_PROCESS.wait()
    else:
        SAY_PROCESS.terminate()

    print(message)
    SAY_PROCESS = subprocess.Popen(["say", "-r", str(SAY_RATE), "-v", VOICE, message])


def on_press(key: str, canvas: tk.Canvas, root: tk.Tk) -> Dict:
    """
    Handles key press events for navigating the accessibility tree and triggering actions.

    Args:
        key (str): The key that was pressed.
        canvas (tk.Canvas): The canvas used for drawing the UI.
        root (tk.Tk): The Tkinter root window.

    Returns:
        Dict: The updated active node.
    """
    global ACTIVE_ELEMENT
    active = ACTIVE_ELEMENT
    old_active = active
    key = key.lower()

    if key == "down" and active.get("children"):
        active = active["children"][0]
    elif key == "up" and active.get("parent"):
        active = active["parent"]
    elif key == "left" and active.get("prev"):
        active = active["prev"]
    elif key == "right" and active.get("next"):
        active = active["next"]
    elif key == "space":
        # Calculate click coordinates
        x = root.winfo_x() + (active["box"][0] + active["box"][2]) // 2
        y = root.winfo_y() + (active["box"][1] + active["box"][3]) // 2

        open_app_in_foreground(BUNDLE_ID)
        mouse.position = (x, y)
        time.sleep(0.5)
        mouse.click(Button.left)
        time.sleep(0.5)

        active = get_accessibility_data(BUNDLE_ID)
        wake_up(root)
    elif key == "q":
        root.destroy()
        os.system("killall say")
        exit(0)
    elif key == "h":
        read_message(HELP_MESSAGE)

    if old_active != active:
        os.system("clear")
        print(active["cls"], active["value"], active["box"])

        canvas.delete("all")
        color = "red" if active["cls"] == "Group" else "green"
        canvas.create_rectangle(*active["box"], outline=color)

        message = get_message(active)
        read_message(message)

    return active


def find_element(element: UIElement, data: UIElement) -> UIElement:
    """
    Find the element in the data tree based on bounding box overlap.

    Args:
        element (UIElement): The element to find.
        data (UIElement): The data tree.

    Returns:
        UIElement: The found element.
    """
    stack = [data]
    while stack:
        node = stack.pop()

        if iou(element["box"], node["box"]) > 0.8:
            return node

        if "children" in node:
            stack.extend(node["children"])

    return data


def map_system_accessibility_to_ui_element(accessibility: dict) -> UIElement:
    stack = [accessibility]

    while stack:
        element = stack.pop(0)

        if element["children"]:
            stack.extend( element["children"] )

        element['cls'] = element.pop("role")

        element['box'] = element.pop("visible_bbox") if "visible_bbox" in element else element.pop("bbox")
        element['box'] = tuple(map(lambda x: 2 * int(x), element['box'])) if element['box'] else (0, 0, 0, 0)

        element['value'] = f"{element['role_description'] or ''} {element['description'] or ''} {element['value'] or ''}"
        element['value'] = element['value'].strip()

        to_delete = set(element.keys()).difference({"cls", "box", "value", "children"})
        for d in to_delete:
            del element[d]

    return accessibility

def get_accessibility_data(bundle_id: str) -> UIElement:
    """
    Generates accessibility data by taking a screenshot of the target app and
    processing it.

    Args:
        bundle_id (str): The bundle ID of the target application.

    Returns:
        Dict: The updated accessibility data.
    """
    global SCREEN, ACTIVE_ELEMENT

    open_app_in_foreground(bundle_id)
    screen_path = screenshot_app(bundle_id, "./screenshots/")[0]
    image = Image.open(screen_path)

    if SCREEN:
        # Check if the screen has changed
        diff = ImageChops.difference(SCREEN.convert("RGB"), image.convert("RGB"))
        if not diff.getbbox():
            return ACTIVE_ELEMENT

    SCREEN = image

    # Generate hierarchy
    if SYSTEM_ACCESSIBILITY:
        data = get_tree(bundle_id)
        data = map_system_accessibility_to_ui_element(data)
    else:
        data = generate_hierarchy(SCREEN).to_dict()

    # add prev, next and parent links
    active = update_accessibility_data(data, n=SKIP_GROUPS_SIZE)

    # Try to find the previously active element on the new screen
    if ACTIVE_ELEMENT:
        active = find_element(ACTIVE_ELEMENT, active)

    return active


def parse_arguments():
    """
    Parses command-line arguments.
    """
    global BUNDLE_ID, WELCOME, HELP, SAY_RATE, SYSTEM_ACCESSIBILITY, VOICE, SKIP_GROUPS_SIZE
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bundle_id", type=str, help="The bundle ID of the target application")
    parser.add_argument("-n", "--name", type=str, help="Name of the target application (alternative to bundle_id)")
    parser.add_argument("-dw", "--deactivate_welcome", action="store_true", help="Deactivate welcome message. Do not read \"Welcome to the ScreenReader.\" at startup.")
    parser.add_argument("-dh", "--deactivate_help", action="store_true", help="Deactivate help message. Do not read the help message at startup.")
    parser.add_argument("-r", "--rate", type=int, default=SAY_RATE, help=f"The speech rate for the 'say' command. Default: {SAY_RATE}")
    parser.add_argument("-v", "--voice", help="The voice to use for the 'say' command. See 'say -v \"?\" | grep en' for a list of available voices.", default=VOICE)
    parser.add_argument("-sa", "--system_accessibility", action="store_true", help="Use system accessibility data instead of generating it")
    parser.add_argument("-sk", "--skip-groups", type=int, default=SKIP_GROUPS_SIZE, help=f"Skip groups with less than n children. Default: {SKIP_GROUPS_SIZE}")
    args = parser.parse_args()

    if args.name and not args.bundle_id:
        BUNDLE_ID = get_app_bundle(args.name)
    else:
        BUNDLE_ID = args.bundle_id

    WELCOME = not args.deactivate_welcome
    HELP = not args.deactivate_help
    SAY_RATE = args.rate
    SYSTEM_ACCESSIBILITY = args.system_accessibility
    VOICE = args.voice
    SKIP_GROUPS_SIZE = args.skip_groups


def main():
    """
    Main entry point for the ScreenReader application.
    """
    parse_arguments()

    global ACTIVE_ELEMENT, SAY_PROCESS, WELCOME, HELP, WELCOME_MESSAGE, HELP_MESSAGE, BUNDLE_ID

    if WELCOME:
        read_message(WELCOME_MESSAGE)

    if HELP:
        read_message(HELP_MESSAGE, wait=True)

    ACTIVE_ELEMENT = get_accessibility_data(BUNDLE_ID)

    app_info = get_app_info(BUNDLE_ID)
    app_name = get_app_name(BUNDLE_ID)
    children_count = len(ACTIVE_ELEMENT.get("children", []))
    ACTIVE_ELEMENT["value"] = (
        f"{app_name}, global window, {app_info[1].replace('_', ' ')}. "
        f"Press down to view {children_count} items."
    )

    root, canvas = create_app(app_info[2])
    canvas.create_rectangle(*ACTIVE_ELEMENT["box"], outline="red")

    def on_press_listener(key: str) -> None:
        global ACTIVE_ELEMENT
        print(key)
        ACTIVE_ELEMENT = on_press(key, canvas, root)

    root.bind("<Key>", lambda event: on_press_listener(event.keysym))
    wake_up(root)

    message = get_message(ACTIVE_ELEMENT)
    read_message(message, wait=True)
    root.mainloop()


if __name__ == "__main__":
    main()
