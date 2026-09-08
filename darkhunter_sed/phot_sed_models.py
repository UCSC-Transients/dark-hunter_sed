"""
Path-2 forward photometry models (1-star in this issue).

Maps free parameters → predicted magnitudes via MISTy (EEP+M) → PHOENIX
atmosphere → F99 R_V=3.1 → synphot registry bands.

Do **not** route through uberMS SVI / Payne photNN.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from darkhunter_sed.misty_iso import MistPoint, MistPredictFn, evaluate_mist
from darkhunter_sed.phoenix_grid import PhoenixGrid, phoenix_synth_phot

# Free-parameter order for 1-star dynesty (locked Path-2 plan).
ONE_STAR_PARAM_NAMES: tuple[str, ...] = (
    "EEP",
    "M",
    "FeH",
    "aFe",
    "Av",
    "parallax",
)


class SynthPhotFn(Protocol):
    """Injectable PHOENIX×F99×synphot callable for tests."""

    def __call__(
        self,
        teff_k: float,
        logg: float,
        mh: float,
        alpha: float,
        a_v: float,
        bands: Sequence[str],
        *,
        radius_cm: float,
        distance_pc: float,
        systems: Sequence[str] = ("ab",),
        bandpasses: Mapping[str, object] | None = None,
    ) -> dict[str, dict[str, float]]:
        ...


@dataclass(frozen=True, slots=True)
class OneStarParams:
    """
    1-star free parameters (Path-2).

    Parameters
    ----------
    eep :
        Equivalent evolutionary point.
    mass :
        Initial mass (M☉).
    feh :
        ``[Fe/H]``.
    afe :
        ``[α/Fe]``.
    a_v :
        Visual extinction (mag); F99 R_V=3.1.
    parallax_mas :
        Parallax in milliarcseconds (``> 0``); distance = 1000/ϖ pc.
    """

    eep: float
    mass: float
    feh: float
    afe: float
    a_v: float
    parallax_mas: float

    @classmethod
    def from_array(cls, theta: Sequence[float] | NDArray[np.floating]) -> OneStarParams:
        """
        Pack a length-6 vector in :data:`ONE_STAR_PARAM_NAMES` order.

        Limits
        ------
        Raises ``ValueError`` if ``len(theta) != 6``.
        """
        arr = np.asarray(theta, dtype=np.float64).ravel()
        if arr.size != len(ONE_STAR_PARAM_NAMES):
            raise ValueError(
                f"1-star theta must have length {len(ONE_STAR_PARAM_NAMES)}, got {arr.size}"
            )
        return cls(
            eep=float(arr[0]),
            mass=float(arr[1]),
            feh=float(arr[2]),
            afe=float(arr[3]),
            a_v=float(arr[4]),
            parallax_mas=float(arr[5]),
        )

    def as_array(self) -> NDArray[np.float64]:
        """Return parameters as a length-6 float64 vector."""
        return np.array(
            [self.eep, self.mass, self.feh, self.afe, self.a_v, self.parallax_mas],
            dtype=np.float64,
        )


def parallax_mas_to_distance_pc(parallax_mas: float) -> float:
    """
    Convert parallax (mas) to heliocentric distance (pc).

    Parameters
    ----------
    parallax_mas :
        Parallax in milliarcseconds; must be ``> 0``.

    Returns
    -------
    float
        ``1000 / parallax_mas``.

    Limits
    ------
    No Lutz–Kelker or zero-point corrections.
    """
    p = float(parallax_mas)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("parallax_mas must be finite and > 0")
    return 1000.0 / p


@dataclass(frozen=True, slots=True)
class OneStarPrediction:
    """
    1-star forward-model outputs.

    Parameters
    ----------
    params :
        Input free parameters.
    mist :
        MISTy-derived atmosphere / radius / age.
    mags :
        ``{band: mag}`` in the requested photometric system (default AB).
    distance_pc :
        Distance used for dilution.
    """

    params: OneStarParams
    mist: MistPoint
    mags: dict[str, float]
    distance_pc: float


def predict_1star_phot(
    params: OneStarParams | Sequence[float] | NDArray[np.floating],
    bands: Sequence[str],
    *,
    mist_predictor: MistPredictFn | None = None,
    mist_nn_path: str | None = None,
    phoenix_grid: PhoenixGrid | None = None,
    synth_phot: SynthPhotFn | None = None,
    systems: Sequence[str] = ("ab",),
    bandpasses: Mapping[str, object] | None = None,
    mag_system: str = "ab",
) -> OneStarPrediction:
    """
    1-star: ``EEP, M, [Fe/H], [α/Fe], Av, ϖ`` → predicted photometry.

    Parameters
    ----------
    params :
        :class:`OneStarParams` or length-6 array.
    bands :
        Registry band names present in the observation.
    mist_predictor, mist_nn_path :
        MISTy injectable / path (see :func:`evaluate_mist`).
    phoenix_grid :
        Real or mocked :class:`PhoenixGrid`. Required unless ``synth_phot`` is set.
    synth_phot :
        Optional injectable end-to-end synth (tests); skips ``phoenix_grid``.
    systems, bandpasses :
        Forwarded to PHOENIX synth.
    mag_system :
        Which key to pull from per-band dicts (``\"ab\"`` or ``\"vega\"``).

    Returns
    -------
    OneStarPrediction
        MISTy point + per-band magnitudes.

    Limits
    ------
    IR BB, 2-star, and WD components are out of scope. Extinction is F99 R_V=3.1
    inside :func:`phoenix_synth_phot` when using the grid path.
    """
    p = params if isinstance(params, OneStarParams) else OneStarParams.from_array(params)
    if p.a_v < 0.0:
        raise ValueError("Av must be >= 0")
    mist = evaluate_mist(
        p.eep,
        p.mass,
        p.feh,
        p.afe,
        predictor=mist_predictor,
        mist_nn_path=mist_nn_path,
    )
    distance_pc = parallax_mas_to_distance_pc(p.parallax_mas)

    if synth_phot is not None:
        raw = synth_phot(
            mist.teff_k,
            mist.logg,
            p.feh,
            p.afe,
            p.a_v,
            bands,
            radius_cm=mist.radius_cm,
            distance_pc=distance_pc,
            systems=systems,
            bandpasses=bandpasses,
        )
    else:
        if phoenix_grid is None:
            raise ValueError("phoenix_grid is required when synth_phot is not provided")
        raw = phoenix_synth_phot(
            phoenix_grid,
            mist.teff_k,
            mist.logg,
            p.feh,
            p.afe,
            p.a_v,
            bands,
            radius_cm=mist.radius_cm,
            distance_pc=distance_pc,
            systems=systems,
            bandpasses=bandpasses,
        )

    sys_key = mag_system.lower()
    mags: dict[str, float] = {}
    for band in bands:
        entry = raw[band]
        if sys_key not in entry:
            raise KeyError(f"Band {band} missing magnitude system {sys_key!r}")
        mags[band] = float(entry[sys_key])

    return OneStarPrediction(params=p, mist=mist, mags=mags, distance_pc=distance_pc)
