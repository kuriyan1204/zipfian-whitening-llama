# original: https://github.com/cl-tohoku/zipfian-whitening/blob/main/src/zipfian_whitening.py

from typing import Optional

import torch
from torchtyping import TensorType as TT


class BaseWhitening:
    def __init__(self, save_whitened_W: bool = False):
        """
        Args:
            save_whitened_W (bool, optional): Whether to save whitened embedding matrix W, which could be very large. Defaults to False.
        """
        self.save_whitened_W = save_whitened_W
        self.is_fitted = False

        self.mu: Optional[TT["num_words"]] = None
        self.transformation_matrix: Optional[TT["num_words", "hidden_dim"]] = None

    def fit(
        self,
        W: TT["num_words", "hidden_dim"],
        p: Optional[TT["num_words"]],
    ):
        raise NotImplementedError

    def transform(
        self, W: TT["num_words", "hidden_dim"]
    ) -> TT["num_words", "hidden_dim"]:
        """
        Apply the whitening transformation to the input embeddings.

        Args:
            W torch tensor of shape (num_words,hidden_dim): input embeddings.

        Returns:
            torch tensor of shape (num_words,hidden_dim): whitened embeddings.
        """
        if not self.is_fitted:
            raise ValueError("Whitening has not been fitted yet.")
        W = W - self.mu
        return W @ self.transformation_matrix

    def fit_transform(
        self,
        W: TT["num_words", "hidden_dim"],
        p: Optional[TT["num_words"]],
    ) -> TT["num_words", "hidden_dim"]:
        """
        Fit the whitening and apply it to the input embeddings.
        """
        self.fit(W, p)
        return self.whitened_W if self.save_whitened_W else self.transform(W)


class ZipfianWhitening(BaseWhitening):
    def __init__(self, save_whitened_W: bool = False):
        super().__init__(save_whitened_W)

    def fit(
        self, W: TT["num_words", "hidden_dim"], p: TT["num_words"]
    ) -> "ZipfianWhitening":
        mu: TT["hidden_dim"] = p @ W
        self.mu = mu
        W_centered: TT["num_words", "hidden_dim"] = W - mu
        Wp: TT["num_words"] = W_centered * torch.sqrt(
            p[:, None]
        )  # TODO: check this out why it's not working

        U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
        S_inv = torch.diag(torch.reciprocal(S))
        V = Vt.T
        self.V = V
        self.S_inv = S_inv
        self.transformation_matrix = V @ S_inv
        self.whitened_W = (
            Wp @ self.transformation_matrix if self.save_whitened_W else None
        )
        self.is_fitted = True

        return self


class UniformWhitening(BaseWhitening):
    def __init__(self, save_whitened_W: bool = False):
        super().__init__(save_whitened_W)

    def fit(
        self, W: TT["num_words", "hidden_dim"], p: TT["num_words"]
    ) -> "UniformWhitening":
        mu: TT["num_words"] = torch.mean(W, axis=0)
        self.mu = mu
        W_centered: TT["num_words", "hidden_dim"] = W - mu
        p = None  # XXX: unigram prob will never be used in uniform whitening

        U, S, Vt = torch.linalg.svd(W_centered, full_matrices=False)
        S_inv = torch.diag(
            torch.reciprocal(S / torch.sqrt(torch.tensor(W.shape[0] - 1)))
        )
        V = Vt.T
        self.V = V
        self.S_inv = S_inv
        self.transformation_matrix = V @ S_inv
        self.whitened_W = (
            W_centered @ self.transformation_matrix if self.save_whitened_W else None
        )
        self.is_fitted = True

        return self
