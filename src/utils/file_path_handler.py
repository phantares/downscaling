from pathlib import Path
from omegaconf import OmegaConf


def add_checkpoint_save_path(path: str, cfg):
    save_path = Path("checkpoints", path)

    checkpoint_save_dict = OmegaConf.create(
        {"trainer": {"fit": {"callbacks": {"checkpoint": {"dirpath": save_path}}}}}
    )

    cfg = OmegaConf.merge(checkpoint_save_dict, cfg)

    return cfg
