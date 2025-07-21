import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

cache_dir = "./.models"

model_path = "macpaw-research/blip-icon-captioning"
processor = BlipProcessor.from_pretrained(model_path, cache_dir=cache_dir)
model = BlipForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir).to(device)
model.eval()

@torch.no_grad()
def generate_captions(images: list[Image.Image]) -> list[str]:
    inputs = processor(images, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    return processor.batch_decode(outputs, skip_special_tokens=True)
