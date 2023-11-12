from typing import List
from app.engine import Value
from app.layers.dense import DenseLayer


class MLP:
    def __init__(self, dim_in: int, dim_out: list):
        self.sz = [dim_in] + dim_out
        self.layers = [
            DenseLayer(self.sz[i], self.sz[i + 1]) for i in range(len(dim_out))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
