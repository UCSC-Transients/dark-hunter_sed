"""Neural-network model path resolution for uberMS."""

from __future__ import annotations

from darkhunter_sed.stellar_data import (
    ensure_stellar_sys_path,
    normalize_phot_nn_dir,
    resolve_models_dir,
    resolve_spec_nn_path,
)


def init_stellar_stack():
    """Add uberMS / Payne / MISTy to sys.path and return stellar root."""
    return ensure_stellar_sys_path()


def model_paths(root=None) -> tuple[str, str, str]:
    """Return (specNN path, photNN dir with trailing slash, mistNN path)."""
    root = init_stellar_stack() if root is None else root
    models = resolve_models_dir(root)
    spec = resolve_spec_nn_path(root)
    phot = normalize_phot_nn_dir(models / "photNN")
    mist = str((models / "mistNN" / "mistyNN_2.3_v256_v0.h5").resolve())
    return spec, phot, mist
