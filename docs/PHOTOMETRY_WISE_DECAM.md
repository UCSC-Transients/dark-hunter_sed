# Photometry gather: WISE + DECam-u

Parent: [UCSC-Transients/dark-hunter_pop#31](https://github.com/UCSC-Transients/dark-hunter_pop/issues/31).

`python -m darkhunter_sed.photometry_gather <source_id>` now gathers:

| Band | Source |
|---|---|
| `WISE_W1`, `WISE_W2` | IRSA AllWISE (`allwise_p3as_psd`) |
| `DECam_u` | SkyMapper DR2 u-band (DECam filter): Gaia `skymapperdr2_best_neighbour` + `skymapperdr2_join`, else Vizier `II/358/dr2` |

Band order for uberMS ingestion: see `darkhunter_sed.stellar_data.PREFERRED_BAND_ORDER`.
