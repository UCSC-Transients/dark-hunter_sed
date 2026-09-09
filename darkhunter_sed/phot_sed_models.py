"""
Path-2 forward photometry models (1-star and 2-star coeval).

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

from darkhunter_sed.misty_iso import (
    DEFAULT_EEP_BOUNDS,
    MistPoint,
    MistPredictFn,
    evaluate_mist,
)
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


# ---------------------------------------------------------------------------
# 2-star coeval model (Issue #31)
# ---------------------------------------------------------------------------

# Free-parameter order for 2-star dynesty (locked Path-2 plan).
TWO_STAR_PARAM_NAMES: tuple[str, ...] = (
    "EEP1",
    "M1",
    "M2",
    "FeH",
    "aFe",
    "Av",
    "parallax",
)


class SynthPhot2StarFn(Protocol):
    """
    Injectable 2-star combined SED callable for tests.

    Receives both stars' atmospheric parameters and returns per-band magnitudes
    for the **combined** SED (flux sum) already including shared extinction.
    """

    def __call__(
        self,
        teff1_k: float,
        logg1: float,
        teff2_k: float,
        logg2: float,
        mh: float,
        alpha: float,
        a_v: float,
        bands: Sequence[str],
        *,
        radius1_cm: float,
        radius2_cm: float,
        distance_pc: float,
        systems: Sequence[str] = ("ab",),
        bandpasses: Mapping[str, object] | None = None,
    ) -> dict[str, dict[str, float]]:
        ...


@dataclass(frozen=True, slots=True)
class TwoStarParams:
    """
    2-star coeval free parameters (Path-2, Issue #31).

    Parameters
    ----------
    eep1 :
        Equivalent evolutionary point of the primary.
    mass1 :
        Primary initial mass (M☉); must be ``>= mass2``.
    mass2 :
        Secondary initial mass (M☉); must be ``<= mass1``.
    feh :
        Shared ``[Fe/H]``.
    afe :
        Shared ``[α/Fe]``.
    a_v :
        Shared visual extinction (mag); F99 R_V=3.1.
    parallax_mas :
        Shared parallax in milliarcseconds (``> 0``).

    Notes
    -----
    EEP₂ (secondary) is not free: it is solved from the coeval age constraint
    ``age(EEP₂, M₂, FeH, aFe) = age(EEP₁, M₁, FeH, aFe)`` inside
    :func:`predict_2star_phot`.
    """

    eep1: float
    mass1: float
    mass2: float
    feh: float
    afe: float
    a_v: float
    parallax_mas: float

    @classmethod
    def from_array(cls, theta: Sequence[float] | NDArray[np.floating]) -> TwoStarParams:
        """
        Pack a length-7 vector in :data:`TWO_STAR_PARAM_NAMES` order.

        Limits
        ------
        Raises ``ValueError`` if ``len(theta) != 7``.
        """
        arr = np.asarray(theta, dtype=np.float64).ravel()
        if arr.size != len(TWO_STAR_PARAM_NAMES):
            raise ValueError(
                f"2-star theta must have length {len(TWO_STAR_PARAM_NAMES)}, got {arr.size}"
            )
        return cls(
            eep1=float(arr[0]),
            mass1=float(arr[1]),
            mass2=float(arr[2]),
            feh=float(arr[3]),
            afe=float(arr[4]),
            a_v=float(arr[5]),
            parallax_mas=float(arr[6]),
        )

    def as_array(self) -> NDArray[np.float64]:
        """Return parameters as a length-7 float64 vector."""
        return np.array(
            [self.eep1, self.mass1, self.mass2, self.feh, self.afe, self.a_v, self.parallax_mas],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class TwoStarPrediction:
    """
    2-star coeval forward-model outputs.

    Parameters
    ----------
    params :
        Input free parameters.
    mist1 :
        MISTy point for the primary.
    mist2 :
        MISTy point for the secondary (EEP solved for age match).
    eep2 :
        Solved EEP of the secondary.
    mags :
        ``{band: mag}`` for the **combined** SED in the requested system.
    distance_pc :
        Distance used for dilution (from shared parallax).
    """

    params: TwoStarParams
    mist1: MistPoint
    mist2: MistPoint
    eep2: float
    mags: dict[str, float]
    distance_pc: float


def solve_eep2_for_age_match(
    target_age_gyr: float,
    mass2: float,
    feh: float,
    afe: float,
    *,
    predictor: MistPredictFn,
    eep_bounds: tuple[float, float] = DEFAULT_EEP_BOUNDS,
    xtol: float = 0.5,
) -> float | None:
    """
    Find EEP₂ such that ``age(EEP₂, mass2, feh, afe) ≈ target_age_gyr``.

    Parameters
    ----------
    target_age_gyr :
        Age to match in Gyr (from the primary's MISTy evaluation).
    mass2 :
        Secondary initial mass (M☉).
    feh, afe :
        Shared chemistry (same as primary).
    predictor :
        MISTy ``getMIST`` callable.
    eep_bounds :
        Search interval ``(eep_lo, eep_hi)``. Default: full MISTy range ``[1, 808]``.
    xtol :
        Brent's method tolerance in EEP units (default 0.5).  Corresponds to
        age residuals of ≲ 0.05–0.1 Gyr for typical stellar tracks.

    Returns
    -------
    float or None
        EEP₂ that satisfies the coeval constraint, or ``None`` if
        ``target_age_gyr`` lies outside the age range of ``mass2`` over
        ``eep_bounds`` (caller should return ``−∞`` likelihood).

    Limits
    ------
    Assumes age is monotonically non-decreasing with EEP for a given mass /
    chemistry (true for MISTy LinNet in the main-sequence to RGB range).
    Uses ``scipy.optimize.brentq``; requires ≈ 15–25 ``predictor`` calls.
    """
    from scipy.optimize import brentq

    eep_lo = float(eep_bounds[0])
    eep_hi = float(eep_bounds[1])

    def _age_residual(eep: float) -> float:
        mist = evaluate_mist(eep, mass2, feh, afe, predictor=predictor)
        return mist.age_gyr - target_age_gyr

    f_lo = _age_residual(eep_lo)
    f_hi = _age_residual(eep_hi)

    # Same sign: target age outside the achievable range for this mass/chem.
    if f_lo * f_hi > 0.0:
        return None

    eep2 = float(brentq(_age_residual, eep_lo, eep_hi, xtol=xtol, full_output=False))
    return eep2


def predict_2star_phot(
    params: TwoStarParams | Sequence[float] | NDArray[np.floating],
    bands: Sequence[str],
    *,
    mist_predictor: MistPredictFn | None = None,
    phoenix_grid: PhoenixGrid | None = None,
    synth_2star: SynthPhot2StarFn | None = None,
    systems: Sequence[str] = ("ab",),
    bandpasses: Mapping[str, object] | None = None,
    mag_system: str = "ab",
    eep2_xtol: float = 0.5,
) -> TwoStarPrediction:
    """
    2-star coeval: ``EEP₁, M₁, M₂, [Fe/H], [α/Fe], Aᵥ, ϖ`` → combined photometry.

    Parameters
    ----------
    params :
        :class:`TwoStarParams` or length-7 array.
    bands :
        Registry band names present in the observation.
    mist_predictor :
        MISTy ``getMIST`` injectable (required).
    phoenix_grid :
        Real :class:`PhoenixGrid` for production. Required unless ``synth_2star`` given.
    synth_2star :
        Optional injectable combined SED callable (tests); skips ``phoenix_grid``.
        Must return ``{band: {"ab": float}}`` for the **combined, extincted** SED.
    systems, bandpasses, mag_system :
        Forwarded to the SED synthesis step.
    eep2_xtol :
        EEP tolerance for the coeval solver (passed to :func:`solve_eep2_for_age_match`).

    Returns
    -------
    TwoStarPrediction
        Both MISTy points, solved EEP₂, and per-band combined magnitudes.

    Raises
    ------
    ValueError
        If ``mass2 > mass1`` (M₁ ≥ M₂ constraint violated), if no coeval EEP₂
        exists in the grid bounds, or if ``phoenix_grid`` is absent when needed.

    Limits
    ------
    Path-2 photometry order: dilute star 1 + dilute star 2 → **sum** on shared
    bandpass-λ grid → **one** F99 application → synthesize mags.  Never stack
    per-component magnitudes.
    """
    from darkhunter_sed.phoenix_grid import phoenix_synth_phot_2star

    p = params if isinstance(params, TwoStarParams) else TwoStarParams.from_array(params)

    if p.mass2 > p.mass1:
        raise ValueError(
            f"mass2 ({p.mass2}) > mass1 ({p.mass1}): M₁ ≥ M₂ constraint violated"
        )
    if p.a_v < 0.0:
        raise ValueError("Av must be >= 0")

    # Primary
    mist1 = evaluate_mist(p.eep1, p.mass1, p.feh, p.afe, predictor=mist_predictor)

    # Solve secondary EEP for coeval constraint
    eep2 = solve_eep2_for_age_match(
        mist1.age_gyr,
        p.mass2,
        p.feh,
        p.afe,
        predictor=mist_predictor,
        xtol=eep2_xtol,
    )
    if eep2 is None:
        raise ValueError(
            f"No coeval EEP₂ found for age {mist1.age_gyr:.4f} Gyr "
            f"at M₂={p.mass2:.3f}, [Fe/H]={p.feh:.2f}, [α/Fe]={p.afe:.2f}"
        )

    # Secondary
    mist2 = evaluate_mist(eep2, p.mass2, p.feh, p.afe, predictor=mist_predictor)

    distance_pc = parallax_mas_to_distance_pc(p.parallax_mas)

    if synth_2star is not None:
        raw = synth_2star(
            mist1.teff_k,
            mist1.logg,
            mist2.teff_k,
            mist2.logg,
            p.feh,
            p.afe,
            p.a_v,
            bands,
            radius1_cm=mist1.radius_cm,
            radius2_cm=mist2.radius_cm,
            distance_pc=distance_pc,
            systems=systems,
            bandpasses=bandpasses,
        )
    else:
        if phoenix_grid is None:
            raise ValueError("phoenix_grid is required when synth_2star is not provided")
        raw = phoenix_synth_phot_2star(
            phoenix_grid,
            mist1,
            mist2,
            p.feh,
            p.afe,
            p.a_v,
            bands,
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

    return TwoStarPrediction(
        params=p,
        mist1=mist1,
        mist2=mist2,
        eep2=float(eep2),
        mags=mags,
        distance_pc=distance_pc,
    )
