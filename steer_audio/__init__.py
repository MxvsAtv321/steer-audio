"""
steer_audio — core library for the TADA project.

Provides:
  - SteeringVector / SteeringVectorBank  (vector_bank)
  - MultiConceptSteerer                  (multi_steer)
  - TimestepAdaptiveSteerer              (temporal_steering)
  - constant_schedule / cosine_schedule  (temporal_steering)
  - early_only_schedule / late_only_schedule / linear_schedule  (temporal_steering)
  - ConceptFeatureSet / ConceptAlgebra   (concept_algebra)
"""

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank  # noqa: F401
from steer_audio.multi_steer import MultiConceptSteerer  # noqa: F401
from steer_audio.temporal_steering import (  # noqa: F401
    TimestepAdaptiveSteerer,
    TimestepSchedule,
    constant_schedule,
    cosine_schedule,
    early_only_schedule,
    late_only_schedule,
    linear_schedule,
)
from steer_audio.concept_algebra import (  # noqa: F401
    ConceptFeatureSet,
    ConceptAlgebra,
    AlgebraPreset,
    AlgebraPresetBank,
)
