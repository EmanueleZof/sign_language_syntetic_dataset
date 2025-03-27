from pathlib import Path

def create_dir(self, path):
    Path(path).mkdir(parents=True, exist_ok=True)