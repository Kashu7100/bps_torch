from strenum import StrEnum


class BPS_TYPE(StrEnum):
    RANDOM_UNIFORM = "random_uniform"
    RANDOM_NONUNIFORM = "random_nonuniform"
    GRID_CUBE = "grid_cube"
    GRID_SPHERE = "grid_sphere"
    CUSTOM = "custom"


class BPS_FEATURE_TYPE(StrEnum):
    DISTS = "dists"
    DELTAS = "deltas"
    CLOSEST = "closest"
    FEATURES = "features"
