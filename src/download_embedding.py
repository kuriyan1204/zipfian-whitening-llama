"""
Download and extract static word embedding layer from large language models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import safetensors.torch
import torch
from fire import Fire
from huggingface_hub import hf_hub_download

from config import get_config
from utils import convert_hf_name_to_local_path_name


def save_embedding_layer(
    model_name: str,
    embedding_layer: torch.Tensor,
) -> None:
    path = convert_hf_name_to_local_path_name(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding_layer, path)


def main(config_name: str):
    config = get_config(config_name)

    safetensor_cache_path = hf_hub_download(
        repo_id=config.hf_model_name,
        filename=config.safetensor_file_name,
    )

    with safetensors.torch.safe_open(safetensor_cache_path, "pt") as f:
        embedding_layer = f.get_tensor(config.embedding_layer_name)

    save_embedding_layer(config.hf_model_name, embedding_layer)


if __name__ == "__main__":
    Fire(main)
