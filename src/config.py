CONFIG_REGISTRY = {}


def register_config(name):
    def decorator(cls):
        CONFIG_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class BaseConfig:
    pass


@register_config("llama2-70b")
class Llama2Config70b(BaseConfig):
    hf_model_name = "meta-llama/Llama-2-70b-hf"
    safetensor_file_name = "model-00001-of-00015.safetensors"
    embedding_layer_name = "model.embed_tokens.weight"
    word_freq_file_path = "data/word_freq/llama2/dolma.json"

    transform_name_to_hf_hub_upload_path = {
        "mean": "hkurita/llama-2-70b-embedding_mean",
        "uniform_centering": "hkurita/llama-2-70b-embedding_mean-uniform-centered",
        "uniform_whitening": "hkurita/llama-2-70b-embedding_mean-uniform-whitened",
        "zipfian_centering": "hkurita/llama-2-70b-embedding_mean-zipfian-centered",
        "zipfian_whitening": "hkurita/llama-2-70b-embedding_mean-zipfian-whitened",
    }


def get_config(name: str) -> BaseConfig:
    config = CONFIG_REGISTRY.get(name.lower())
    if config is None:
        raise ValueError(
            f"Config name {name} is not supported. \n Supported configs: {list(CONFIG_REGISTRY.keys())}"
        )
    return config
