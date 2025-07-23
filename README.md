[![MacPaw Research](https://pbs.twimg.com/profile_banners/3993798502/1720615716/1500x500)](https://research.macpaw.com)

# Screen2AX

A research-driven project for generating accessibility of macOS applications using computer vision and deep learning. Read more about the project in our [paper](https://arxiv.org/abs/2507.16704).

---

## 📁 Datasets

- [Screen2AX-Tree](https://huggingface.co/datasets/MacPaw/Screen2AX-Tree)
- [Screen2AX-Element](https://huggingface.co/datasets/MacPaw/Screen2AX-Element)
- [Screen2AX-Group](https://huggingface.co/datasets/MacPaw/Screen2AX-Group)
- [Screen2AX-Task](https://huggingface.co/datasets/MacPaw/Screen2AX-Task)

## 🤖 Models

- [YOLOv11l — UI Elements Detection](https://huggingface.co/MacPaw/yolov11l-ui-elements-detection)  
- [BLIP — UI Elements Captioning](https://huggingface.co/MacPaw/blip-icon-captioning)
- [YOLOv11l — UI Groups Detection](https://huggingface.co/MacPaw/yolov11l-ui-groups-detection)  

---

## 🛠 Requirements

- macOS  
- Python (recommended ≥ 3.11)
- Conda  
- Pip  

---

## ⚙️ Installation

Create and activate the project environment:

```bash
conda create -n screen2ax python=3.11
conda activate screen2ax
pip install -r requirements.txt
```

## 🚀 Usage

> ⚠️ The first run may take longer due to model downloads and initial setup.

### Accessibility generation
Run the accessibility generation script:

```bash
python -m hierarchy_dl.hierarchy --help
```
#### Available Options

```
usage: hierarchy.py [-h] [--image IMAGE] [--save] [--filename FILENAME] [--save_dir SAVE_DIR] [--flat]

options:
  -h, --help           show this help message and exit
  --image IMAGE        Path to the image
  --save               Save the result
  --filename FILENAME  Filename to save the result
  --save_dir SAVE_DIR  Directory to save the result. Default is './results/'
  --flat               Generate flat hierarchy (no groups)
```

##### Example
Run the accessibility generation script on a screenshot of the Spotify app:

```bash
python -m hierarchy_dl.hierarchy --image ./screenshots/spotify.png --save --filename spotify.json
```

This will generate a JSON file with the accessibility of the app in the results folder.

### Screen Reader 
Run the screen reader:

```bash
python -m screen_reader.screen_reader --help
```

#### Available Options

```
usage: screen_reader.py [-h] [-b BUNDLE_ID] [-n NAME] [-dw] [-dh] [-r RATE] [-v VOICE] [-sa] [-sk SKIP_GROUPS]

options:
  -h, --help                    show this help message and exit
  -b, --bundle_id BUNDLE_ID     Bundle ID of the target application
  -n, --name NAME               Name of the target application (alternative to bundle_id)
  -dw, --deactivate_welcome     Skip the "Welcome to the ScreenReader." message
  -dh, --deactivate_help        Skip reading the help message on startup
  -r, --rate RATE               Set speech rate for macOS `say` command (default: 190)
  -v, --voice VOICE             Set voice for macOS `say` command (see `say -v "?" | grep en`)
  -sa, --system_accessibility   Use macOS system accessibility data instead of vision-generated
  -sk, --skip-groups N          Skip groups with fewer than N children (default: 5)
```

##### Example

Run the screen reader for the Spotify app:
```bash
python -m screen_reader.screen_reader --name Spotify
```

## 📜 License
### 🔍 YOLO Models
The YOLO models used for UI elements and UI groups detection are licensed under the GNU Affero General Public License (AGPL). This is inherited from the original YOLO model licensing.

### 🧠 BLIP Model
The BLIP model for captioning UI elements is provided under the MIT License. 

### 📂 Datasets
All datasets (Screen2AX-Tree, Screen2AX-Element, Screen2AX-Group, Screen2AX-Task) are released under the Apache 2.0 license.

### 💻 Codebase
All source code in this repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for full terms and conditions.

## 📚 Citation
If you use this code in your research, please cite our paper:

```bibtex
@misc{muryn2025screen2axvisionbasedapproachautomatic,
      title={Screen2AX: Vision-Based Approach for Automatic macOS Accessibility Generation}, 
      author={Viktor Muryn and Marta Sumyk and Mariya Hirna and Sofiya Garkot and Maksym Shamrai},
      year={2025},
      eprint={2507.16704},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.16704}, 
}
```

## 🙌 Acknowledgements
We would like to express our deepest gratitude to the Armed Forces of Ukraine. Your courage and unwavering defense of our country make it possible for us to live, work, and create in freedom. This work would not be possible without your sacrifice. Thank you.

## MacPaw Research

Visit our site to learn more 😉

https://research.macpaw.com
