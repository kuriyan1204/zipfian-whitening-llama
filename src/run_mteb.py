"""
Run MTEB benchmark to pre-trained embeddings.
TODO: evaluate with all tasks on MTEB
"""

import mteb
from sentence_transformers import SentenceTransformer

model_names = [
    "hkurita/llama-2-70b-embedding_mean",
    "hkurita/llama-2-70b-embedding_mean-uniform-centered",
    "hkurita/llama-2-70b-embedding_mean-uniform-whitened",
    "hkurita/llama-2-70b-embedding_mean-zipfian-centered",
    "hkurita/llama-2-70b-embedding_mean-zipfian-whitened",
]
for model_name in model_names:
    model = SentenceTransformer(model_name)
    tasks = [
        "STSBenchmark",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16", 
    ]

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"results/{model_name.replace('/','---')}")