import json
import jsonschema
import joblib
from pathlib import Path
from types import SimpleNamespace
from trainer import Trainer
from utils.seed_utils import set_seed
from utils.experiment_utils import get_next_version

def check_and_get_configuration(filename: str, validation_filename: str) -> object:
    """Valida un file di configurazione JSON con un file schema JSON e lo restituisce come oggetto."""
    config_file = Path(filename)
    schema_file = Path(validation_filename)

    if not (config_file.is_file() and schema_file.is_file() and
        config_file.suffix == '.json' and schema_file.suffix == '.json'):
        print("Config or schema file not found.")
        return None

    with config_file.open() as d, schema_file.open() as s:
        try:
            data = json.load(d)
            schema = json.load(s)
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            print(f"[CONFIG ERROR] Validation failed: {e.message}")
            return None
        except jsonschema.exceptions.SchemaError as e:
            print(f"[SCHEMA ERROR] Invalid schema: {e.message}")
            return None
        
    with config_file.open() as d:
        json_object = json.loads(d.read(), object_hook=lambda d: SimpleNamespace(**d))

    return json_object

def main():
    CONFIG = "./config/config.json"
    SCHEMA = "./config/config_schema.json"

    cfg = check_and_get_configuration(CONFIG, SCHEMA)
    if cfg is None:
        print("Invalid configuration. Aborting.")
        return
    
    set_seed(42)

    scaler_target = joblib.load("scaler_target.pkl")
    
    model_name = cfg.train_parameters.network_type.lower()
    base_exp_path = Path(cfg.io.out_folder)
    model_dir = base_exp_path / model_name
    
    version = get_next_version(model_dir)
    print(f"Starting experiment version: {version}")
    
    trainer = Trainer(cfg, scaler_target=scaler_target, version=version, base_exp_path=str(base_exp_path))
        
    if cfg.parameters.train:
        trainer.train()

    if cfg.parameters.test:
        trainer.test(print_loss=True)
        
    trainer.writer.close()


if __name__ == "__main__":
    main()