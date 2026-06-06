from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest


class MiniAnnData:
    """Small AnnData-like object used to test the library without optional dependencies."""

    def __init__(
        self,
        x: object,
        obs: Optional[pd.DataFrame] = None,
        var_names: Optional[list[str]] = None,
    ) -> None:
        self.X = x
        n_rows, n_columns = np.asarray(x).shape
        self.obs = obs if obs is not None else pd.DataFrame(index=np.arange(n_rows))
        names = var_names if var_names is not None else [str(index) for index in range(n_columns)]
        self.var = pd.DataFrame(index=pd.Index(names))
        self.layers: dict[str, object] = {}
        self.uns: dict[str, object] = {}
        self.obsm: dict[str, object] = {}

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    @var_names.setter
    def var_names(self, values: list[str]) -> None:
        self.var.index = pd.Index(values)

    def copy(self) -> "MiniAnnData":
        copied = MiniAnnData(
            self.X.copy() if hasattr(self.X, "copy") else self.X,
            obs=self.obs.copy(),
            var_names=self.var_names.astype(str).tolist(),
        )
        copied.layers = {key: value.copy() if hasattr(value, "copy") else value for key, value in self.layers.items()}
        copied.uns = dict(self.uns)
        copied.obsm = {key: value.copy() if hasattr(value, "copy") else value for key, value in self.obsm.items()}
        return copied


@pytest.fixture
def make_adata() -> Callable[..., MiniAnnData]:
    def factory(
        x: object,
        obs: Optional[pd.DataFrame] = None,
        var_names: Optional[list[str]] = None,
    ) -> MiniAnnData:
        return MiniAnnData(x=x, obs=obs, var_names=var_names)

    return factory
