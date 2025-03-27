import requests

from pathlib import Path

def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def flush_ram():
    gc.collect()

def download_file(url, path):
    file_name = url.split("/")[-1]
    with open(f"{path}{file_name}", "wb") as handle:
        response = requests.get(url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

        handle.write(block)