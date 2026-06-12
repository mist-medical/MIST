"""Initialize and register all available label ensemblers."""

# Import label ensembler implementations to trigger registration decorators.
from .staple import STAPLEEnsembler  # noqa: F401
from .majority_vote import MajorityVoteEnsembler  # noqa: F401
