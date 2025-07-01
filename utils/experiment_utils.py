from pathlib import Path
import shutil
import re

def create_experiment_dir(base_path, model_name, version, config_path):
    """
    base_path: Path base dove mettere tutti gli esperimenti
    model_name: nome modello in minuscolo
    version: stringa versione (es. "v1", "v2", ...)
    config_path: path al file config originale
    
    Ritorna: Path alla cartella esperimento creata
    """
    model_dir = Path(base_path) / model_name
    exp_dir = model_dir / version
    exp_dir.mkdir(parents=True, exist_ok=True)

    dest_config_path = exp_dir / "config.json"
    if not dest_config_path.exists():
        shutil.copy(config_path, dest_config_path)

    return exp_dir


def get_next_version(base_dir: Path) -> str:
    """
    Cerca le versioni esistenti in base_dir e ritorna la versione successiva "vN".
    Se non ci sono versioni, ritorna "v1".
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return "v1"
    
    versions = []
    pattern = re.compile(r"v(\d+)$")
    for folder in base_dir.iterdir():
        if folder.is_dir():
            match = pattern.fullmatch(folder.name)
            if match:
                versions.append(int(match.group(1)))
    if not versions:
        return "v1"
    return f"v{max(versions) + 1}"