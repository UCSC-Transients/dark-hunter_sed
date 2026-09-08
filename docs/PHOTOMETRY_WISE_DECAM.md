# Photometry gather: WISE + DECam-u

Parent: [UCSC-Transients/dark-hunter_pop#31](https://github.com/UCSC-Transients/dark-hunter_pop/issues/31).

`python -m darkhunter_sed.photometry_gather <source_id>` now gathers:

| Band | Source |
|---|---|
| `WISE_W1`, `WISE_W2` | IRSA AllWISE (`allwise_p3as_psd`) |
| `DECam_u` | SkyMapper DR2 u-band (DECam filter): Gaia `skymapperdr2_best_neighbour` + `skymapperdr2_join`, else Vizier `II/358/dr2` |

W3/W4 are **not** gathered. For Path-2 (PHOENIX HiRes → synphot), W1/W2 thruputs still extend redward of the PHOENIX WAVE file (~5.5 μm), so mid-IR integrals are incomplete until a spectrum extension is added; examine later.

Band order for uberMS ingestion: see `darkhunter_sed.stellar_data.PREFERRED_BAND_ORDER`.
