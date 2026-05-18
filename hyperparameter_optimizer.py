from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class HyperparameterOptimizer(ABC):
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        raise NotImplementedError
