# Dataset Provenance and Evaluation Protocol

This repository does not introduce a new dataset. The local `Unified Road Anomaly Evaluation Set` is a deterministic convenience package assembled from public validation data already downloaded from three sources.

| Source | Local pairs | Role | Standardized anomaly label |
|---|---:|---|---:|
| SMIYC RoadObstacle21 | 30 | Real road-obstacle validation | 1 |
| RoadAnomaly / RoadAnomaly21 subset | 10 | Real anomaly validation | 1 |
| StreetHazards partial | 149 | Synthetic cross-domain stress test | original class 14 -> 1 |

The 189 pairs are not the complete official benchmark test sets. Results in this repository are local controlled-comparison results and must not be described as official leaderboard scores.

## Standardized labels

All exported label PNGs use:

```text
0   normal / in-distribution evaluation pixel
1   anomaly / obstacle
255 ignore / void
```

SMIYC labels contain large ignored regions outside the drivable evaluation area. Those pixels must remain `255` and are excluded from every metric. StreetHazards uses semantic class `14` as the anomaly class in the source labels.

## Sources

- SegmentMeIfYouCan benchmark and RoadObstacle21: https://segmentmeifyoucan.com/
- RoadObstacle21 data record: https://zenodo.org/records/5281633
- StreetHazards official code/data page: https://github.com/hendrycks/anomaly-seg

RoadObstacle21 is published under CC BY 4.0 according to the benchmark documentation. RoadAnomaly images have per-image source/license metadata in the official data record. StreetHazards is distributed by its authors; users should consult the official repository and applicable terms before redistributing the image files.

For the safest public release, publish the code, metadata, checksums, and download/preparation scripts. The LFS archive is a convenience mirror and does not replace the original dataset citations or licenses.

## Rebuild

After obtaining the source datasets in the documented local folders:

```powershell
python scripts\build_unified_dataset.py
python scripts\package_unified_dataset.py
```

The package contains `images/`, `gt_labels/`, and `metadata/`.
