"""CoSpaDi paper's original K-SVD method configuration."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MethodSpec:
    name: str
    iterations: int
    power_iterations: int = 8
    ks_ratio: float = 2.0
    extra_args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name != 'ksvd':
            raise ValueError('This repository implements only CoSpaDi (ksvd).')
        if self.iterations <= 0 or self.power_iterations <= 0 or self.ks_ratio <= 0:
            raise ValueError('iterations, power_iterations and ks_ratio must be positive')
