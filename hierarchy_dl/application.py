import time
import threading

import tkinter as tk

from hierarchy import generate_hierarchy
from screen_reader.screenshot import screenshot_app, open_app_in_foreground


run = True
thread = None


def start_action():
    bundle_id = entry.get()
    open_app_in_foreground(bundle_id, wait_time=2)

    global run
    run = True

    i = 0
    while run:
        try:
            start = time.time()
            open_app_in_foreground(bundle_id, wait_time=0.25)
            screen_path = screenshot_app(bundle_id, f"./screenshots/")[0]

            tree = generate_hierarchy(screen_path, save=True, save_dir=f"./result/{bundle_id}/")

            end = time.time()

            i += 1
            print(f"Frame #{i}, time taken: {end - start}")

        except Exception as e:
            print(f"Error: {e}")
            break


def stop_action():
    global run, thread
    run = False
    print(f"Stopping process")

    if thread:
        thread.join()

    print(f"Thread has stopped")


def start_thread():
    global thread
    thread = threading.Thread(target=start_action, daemon=True)
    thread.start()


if __name__ == "__main__":
    # Create main window
    root = tk.Tk()
    root.title("Bundle ID Manager")
    root.geometry("300x200")

    # Create input field
    label = tk.Label(root, text="bundle_id:")
    label.pack(pady=5)

    entry = tk.Entry(root)
    entry.pack(pady=5)

    # Copyable text with suggestion
    suggestion = tk.Label(root, text="osascript -e 'id of app \"Spotify\"' \n e.g. com.spotify.client")
    suggestion.pack(pady=5)

    # Create buttons
    start_button = tk.Button(root, text="Start", command=start_thread)
    start_button.pack(pady=5)

    stop_button = tk.Button(root, text="Stop", command=stop_action)
    stop_button.pack(pady=5)

    # Run application
    root.mainloop()