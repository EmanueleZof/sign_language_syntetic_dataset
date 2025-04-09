import gc
import requests

from pathlib import Path

OUTPUT_DIR = "outputs/"

def create_dir(path):
    """
    Crea una cartella (e le cartelle intermedie) nel percorso specificato.
    
    Args:
        path (str): Il percorso in cui creare la cartella.
        
    Returns:
        None
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def flush_ram():
    """
    Libera la memoria RAM non più utilizzata chiamando il garbage collector.
    
    Returns:
        None
    """
    gc.collect()

def download_file(url, path):
    """
    Scarica un file dal link URL e lo salva nel percorso specificato.
    
    Args:
        url (str): URL del file da scaricare.
        path (str): Percorso in cui salvare il file scaricato.
        
    Returns:
        str: Il percorso completo dove il file è stato salvato.
    """
    file_name = url.split("/")[-1]
    save_as = f"{path}{file_name}"
    img_data = requests.get(url).content

    with open(save_as, "wb") as handler:
        handler.write(img_data)

    return save_as