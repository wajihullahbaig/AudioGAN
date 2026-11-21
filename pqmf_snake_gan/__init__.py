"""Multi-Band PQMF + Snake-Activation Conditional Audio GAN (pure time-domain generator)."""

from .pqmf import PQMF
from .modules import (
    SnakeBeta,
    FiLM1d,
    StyleGate1d,
    TemporalSelfAttention1d,
    PhaseShuffle1d,
    DiffAugment1d,
    EMA,
)
from .models import (
    PQMFGenerator,
    CompositeDiscriminator,
    MultiResolutionSTFTDiscriminator,
    MultiPeriodDiscriminator,
    MultiBandDiscriminator,
    default_upsample_rates,
)
from .dataset import AudioMNISTDataset, build_dataset_cache

__all__ = [
    "PQMF",
    "SnakeBeta",
    "FiLM1d",
    "StyleGate1d",
    "TemporalSelfAttention1d",
    "PhaseShuffle1d",
    "DiffAugment1d",
    "EMA",
    "PQMFGenerator",
    "CompositeDiscriminator",
    "MultiResolutionSTFTDiscriminator",
    "MultiPeriodDiscriminator",
    "MultiBandDiscriminator",
    "default_upsample_rates",
    "AudioMNISTDataset",
    "build_dataset_cache",
]
