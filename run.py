import json
import jsonschema
from pathlib import Path
from types import SimpleNamespace
from trainer import Trainer

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

    trainer = Trainer(cfg)
        
    if cfg.parameters.train:
        trainer.train()

    if cfg.parameters.test:
        trainer.test(print_loss=True)


if __name__ == "__main__":
    main()