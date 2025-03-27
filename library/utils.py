import gc
import requests

from pathlib import Path

def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def flush_ram():
    gc.collect()

def download_file(url, path):
    file_name = url.split("/")[-1]
    save_as = f"{path}{file_name}"
    img_data = requests.get(url).content

    with open(save_as, "wb") as handler:
        handler.write(img_data)

    return save_as