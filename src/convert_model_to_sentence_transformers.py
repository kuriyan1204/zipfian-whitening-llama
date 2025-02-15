"""
Apply centering/whitening transformation to the embedding & upload SentenceTransformer model to Hugging Face Hub.
"""

import json
from dataclasses import dataclass
from typing import Optional, Type, Union

import torch
from fire import Fire
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from torchtyping import TensorType as TT
from tqdm import tqdm
from transformers import AutoTokenizer

from config import BaseConfig, get_config
from utils import convert_hf_name_to_local_path_name
from zipfian_whitening.whitening import UniformWhitening, ZipfianWhitening


@dataclass(frozen=True)
class TransformConfig:
    whitening_class: Optional[Type[Union[UniformWhitening, ZipfianWhitening]]]
    do_centering: bool
    do_whitening: bool


TRANSFORM_CONFIGS = {
    "mean": TransformConfig(
        whitening_class=None,
        do_centering=False,
        do_whitening=False,
    ),
    "uniform_centering": TransformConfig(
        whitening_class=UniformWhitening,
        do_centering=True,
        do_whitening=False,
    ),
    "uniform_whitening": TransformConfig(
        whitening_class=UniformWhitening,
        do_centering=True,
        do_whitening=True,
    ),
    "zipfian_centering": TransformConfig(
        whitening_class=ZipfianWhitening,
        do_centering=True,
        do_whitening=False,
    ),
    "zipfian_whitening": TransformConfig(
        whitening_class=ZipfianWhitening,
        do_centering=True,
        do_whitening=True,
    ),
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_unigram_prob(word_freq_file_path: str, vocab_size: int) -> TT["vocab_size"]:
    """
    Load unigram probability from the word frequency file.
    """
    unigram_prob: TT["vocab_size"] = torch.zeros(vocab_size)
    with open(word_freq_file_path, "r") as f:
        unigram_freq: dict[str, int] = json.load(f)
    for vocab_id, freq in unigram_freq.items():
        unigram_prob[int(vocab_id)] = freq
    unigram_prob /= unigram_prob.sum()

    return unigram_prob


def transform_embedding(
    embedding: torch.Tensor, config: BaseConfig, transform_config: TransformConfig
):
    """
    Apply centering/whitening transformation to the embedding.
    TODO: remove unused vocabs from unigram_prob & embedding to reduce noise in whitening (especially, reserved special tokens could be problematic)
    TODO: handle special tokens
    """
    device = get_device()

    if transform_config.whitening_class is None or not (
        transform_config.do_centering or transform_config.do_whitening
    ):
        return embedding
    embedding = embedding.to(device).float()
    vocab_size, _ = embedding.shape
    whitening_transformer = transform_config.whitening_class()
    unigram_prob = get_unigram_prob(config.word_freq_file_path, vocab_size).to(
        device
    )  # XXX: torch svd doesn't support half precision

    whitening_transformer.fit(embedding, unigram_prob)

    if transform_config.do_centering:
        embedding = embedding - whitening_transformer.mu

    if transform_config.do_whitening:
        embedding = embedding @ whitening_transformer.transformation_matrix

    return embedding.half()


def main(config_name: str):
    config = get_config(config_name)
    saved_embedding_path = convert_hf_name_to_local_path_name(config.hf_model_name)
    embedding = torch.load(saved_embedding_path)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)

    for transform_name, transform_config in tqdm(
        TRANSFORM_CONFIGS.items(), desc="Apply centering/whitening & upload to hf hub"
    ):
        transformed_embedding = transform_embedding(embedding, config, transform_config)
        model = SentenceTransformer(
            modules=[
                StaticEmbedding(
                    tokenizer=tokenizer, embedding_weights=transformed_embedding
                )
            ]
        )
        model.push_to_hub(
            repo_id=config.transform_name_to_hf_hub_upload_path[transform_name],
        )


if __name__ == "__main__":
    Fire(main)