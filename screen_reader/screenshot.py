import os
import time
import Quartz
import AppKit
import subprocess
from typing import Iterable, List, Dict, AnyStr, Union, Iterator, Tuple

class ScreencaptureEx(Exception):
    pass

WindowInfo = Dict[AnyStr, Union[AnyStr, int]]

USER_OPTS_STR = "exclude_desktop on_screen_only"
FILE_EXT = "png"
COMMAND = 'screencapture {options} -l {window} -o "{filename}"'
SUCCESS = 0
STATUS_BAR_WINDOW_IDENTIFIER = "Item-0"


def get_window_info() -> List[WindowInfo]:
    return Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionAll
        | Quartz.kCGWindowListExcludeDesktopElements
        | Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
    )


def gen_ids_from_info(
        windows: Iterable[WindowInfo],
) -> List[Tuple[int, str, str]]:  # Changed return type to List
    result = []  # Initialize a list to store results
    for win_dict in windows:
        owner = win_dict.get("kCGWindowOwnerName", "")
        num = win_dict.get("kCGWindowNumber", "")
        name = win_dict.get("kCGWindowName", "")
        bounds = win_dict.get('kCGWindowBounds', "")

        x = bounds['X']
        y = bounds['Y']
        width = bounds['Width']
        height = bounds['Height']

        result.append((num, owner, name, (x, y, width, height)))
    return result


def gen_window_ids(
        parent: str,
) -> List[Tuple[int, str]]:  # Changed return type to List[Tuple[int, str]]
    windows = get_window_info()
    parent = parent.lower()
    result = []  # Initialize a list to store results

    for num, owner, window_name, (x, y, width, height) in gen_ids_from_info(windows):
        if parent == owner.lower():
            if window_name == STATUS_BAR_WINDOW_IDENTIFIER:
                print(f"Skipping status bar window: {num}")
            else:
                window_name = window_name.replace(" ", "_")
                result.append((num, window_name, (x, y, width, height)))

    return result  # Return the list of window IDs


def take_screenshot(window: int, filename: str, output_folder: str) -> str:
    filename = os.path.join(output_folder, filename)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = COMMAND.format(window=window, filename=filename, options="")
    rc, output = subprocess.getstatusoutput(command)
    if rc != SUCCESS:
        raise ScreencaptureEx(f"Error: screencapture output: {output}")

    return filename


def get_filename(window_name, extension) -> str:
    return f"{window_name}:{time.time():.2f}.{extension}"


def gen_windows(application_name: str) -> Iterator[int]:
    windows = list(gen_window_ids(application_name))  # Convert generator to list
    if not windows:  # Check if the list is empty
        print(f"Window with parent {application_name} not found.")
    return windows  # Return the list of windows


def screenshot_windows(
    app_name: str,
    output_folder: str,
    extension: str = "",
) -> Iterator[str]:
    windows = gen_windows(app_name)

    for window_identifier, window_name, _ in windows:
        yield take_screenshot(
            window_identifier, get_filename(window_name, extension), output_folder
        )


def screenshot_application_windows(name: str, output_folder: str, extension: str) -> List[str]:
    filenames = []
    for filename in screenshot_windows(name, output_folder, extension):
        filenames.append(filename)
    return filenames


def running_app(app_bundle):
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    for app in workspace.runningApplications():
        if app.bundleIdentifier() == app_bundle:
            return app
    return None


def screenshot_app(app_bundle: str, output_folder: str) -> List[str]:
    assert app_bundle is not None, "Application bundle is not specified"
    assert output_folder is not None, "Output folder is not specified"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    app = running_app(app_bundle)
    return screenshot_application_windows(app.localizedName(), output_folder, FILE_EXT)


# def get_window_position(processIdentifier):
#     window_list = Quartz.CGWindowListCopyWindowInfo(
#         Quartz.kCGWindowListOptionOnScreenOnly,
#         Quartz.kCGNullWindowID
#     )

#     for window in window_list:
#         if window['kCGWindowOwnerPID'] == processIdentifier:
#             return window['kCGWindowBounds']['X'], window['kCGWindowBounds']['Y']
    
#     return None


def open_app_in_foreground(app_bundle: str, wait_time: float = 2):
    os.system(f"open -b {app_bundle}")
    time.sleep(wait_time)


def get_app_name(bundle_id: str):
    app = running_app(bundle_id)
    return app.localizedName()

def get_app_info(bundle_id: str):
    app = get_app_name(bundle_id)
    return gen_windows(app)[-1]
