from pathlib import Path


def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "lr": 1e-4,
        "seq_len": 128,
        "d_model": 256,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str) -> str:
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    weights_files = list(Path(model_folder).glob(f"{config['model_basename']}*.pt"))

    if not weights_files:
        return None

    weights_files.sort()
    return str(weights_files[-1])