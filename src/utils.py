import time
from contextlib import contextmanager
from pathlib import Path


def convert_hf_name_to_local_path_name(hf_name: str) -> Path:
    dir_name = Path("data/embedding") / hf_name.replace("/", "---")
    return dir_name / "original.pt"


@contextmanager
def timer(label: str = "Elapsed time", verbose: bool = True):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if verbose:
            print(f"{label}: {elapsed:.4f} seconds")
