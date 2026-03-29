# CoreSlicer Reference Values for 4 Patients

## Patient P-0 (IM106)
| Metric | CoreSlicer | Our App | Delta |
|--------|-----------|---------|-------|
| Left Psoas | 7.06 cm² | 0.00 | -7.06 |
| Right Psoas | 7.16 cm² | 0.00 | -7.16 |
| Total Muscle | 103.63 cm² | SMA=96.71 | -6.92 |
| Subcutaneous Fat | 296.30 cm² | N/A | N/A |
| Visceral Fat | 112.36 cm² | N/A | N/A |
| NAMA | N/A | 83.00 | N/A |
| LAMA | N/A | 13.70 | N/A |
| IMAT | N/A | 0.18 | N/A |

## Patient P-1 (IM72)
| Metric | CoreSlicer | Our App | Delta |
|--------|-----------|---------|-------|
| Left Psoas | 8.21 cm² | 0.00 | -8.21 |
| Right Psoas | 4.40 cm² | 0.00 | -4.40 |
| Total Muscle | 82.47 cm² | SMA=45.21 | -37.26 (55% of ref!) |
| Subcutaneous Fat | 110.13 cm² | N/A | N/A |
| Visceral Fat | 83.14 cm² | N/A | N/A |
| NAMA | N/A | 13.21 | N/A |
| LAMA | N/A | 32.00 | N/A |
| IMAT | N/A | 1.82 | N/A |

## Patient P-2 (IM79)
| Metric | CoreSlicer | Our App | Delta |
|--------|-----------|---------|-------|
| Left Psoas | 11.01 cm² | 0.00 | -11.01 |
| Right Psoas | 6.16 cm² | 0.00 | -6.16 |
| Total Muscle | 113.56 cm² | SMA=84.42 | -29.14 (74% of ref) |
| Subcutaneous Fat | 222.21 cm² | N/A | N/A |
| Visceral Fat | 92.69 cm² | N/A | N/A |
| NAMA | N/A | 49.66 | N/A |
| LAMA | N/A | 34.75 | N/A |
| IMAT | N/A | 1.81 | N/A |

## Patient P-3 (unknown slice)
| Metric | CoreSlicer | Our App | Delta |
|--------|-----------|---------|-------|
| Left Psoas | 10.21 cm² | 0.00 | -10.21 |
| Right Psoas | 10.16 cm² | 0.00 | -10.16 |
| Total Muscle | 109.61 cm² | SMA=117.95 | +8.34 (108% of ref) |
| Subcutaneous Fat | 179.21 cm² | N/A | N/A |
| Visceral Fat | 113.35 cm² | N/A | N/A |
| NAMA | N/A | 66.97 | N/A |
| LAMA | N/A | 50.98 | N/A |
| IMAT | N/A | 5.11 | N/A |

## Summary of Discrepancies

### 1. Psoas: COMPLETE FAILURE (0.00 for all 4 patients)
CoreSlicer shows bilateral psoas ranging from 12.61-20.37 cm² total.
Our app detects 0.00 for every patient.

### 2. Total Muscle (SMA): HIGHLY VARIABLE
- P-0: 96.71 vs 103.63 (93% — closest, but still missing psoas)
- P-1: 45.21 vs 82.47 (55% — CATASTROPHIC undercount)
- P-2: 84.42 vs 113.56 (74% — significant undercount)
- P-3: 117.95 vs 109.61 (108% — OVERCOUNTING, likely including visceral tissue)

### 3. Root Cause Analysis
Looking at the CoreSlicer images vs our app:
- CoreSlicer's RED region covers the FULL muscle ring including deep posterior muscles
- Our app only captures a thin peripheral band — missing the thick posterior muscles
  (erector spinae, quadratus lumborum, psoas)
- P-3 overcounts because some visceral tissue leaks into the muscle mask
- P-1 severely undercounts because the patient is thin and the peripheral band
  approach misses most of the muscle

### 4. KEY INSIGHT from CoreSlicer images
CoreSlicer's "Total muscle" (red) covers ALL skeletal muscle at L3:
- Rectus abdominis (anterior)
- Obliques + transversus (lateral, forming the abdominal wall)
- Erector spinae (posterior, THICK group flanking the spine)
- Quadratus lumborum (posterolateral)
- Psoas (anterolateral to vertebra)
The muscle forms a COMPLETE RING, and CoreSlicer captures ALL of it.
Our app only captures a thin outer shell.
