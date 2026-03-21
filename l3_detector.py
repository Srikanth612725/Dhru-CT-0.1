"""
Automatic L3 Vertebral Level Detector
======================================
Identifies the L3 (Third Lumbar Vertebra) slice from a full CT DICOM series
using anatomical landmark detection.

The L3 level is the clinical gold standard for body composition analysis
because single-slice muscle area at L3 correlates best with total body
muscle mass (Mourtzakis et al. 2008, Shen et al. 2004).

Detection Algorithm:
1. Find the iliac crest (top) — corresponds to L4/L5 disc space
2. Find the lowest rib-bearing vertebra — corresponds to T12
3. Estimate L3 as the appropriate fraction between these landmarks
4. Validate using psoas muscle features and SMA profile

Clinical References:
- Mourtzakis M et al. Appl Physiol Nutr Metab 2008
- Shen W et al. Am J Clin Nutr 2004
- Schweitzer L et al. Am J Clin Nutr 2015

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from l3_analyzer import (
    HUThresholds,
    TissueAreas,
    BodyMaskGenerator,
    MuscleCompartmentGenerator,
    TissueSegmenter,
    AreaCalculator,
)


@dataclass
class L3DetectionResult:
    """Result of automatic L3 vertebral level detection."""
    l3_slice_index: int
    l3_slice_location: Optional[float]
    confidence: float  # 0.0 to 1.0
    method: str  # description of detection method used
    iliac_crest_index: Optional[int] = None
    lowest_rib_index: Optional[int] = None
    l3_areas: Optional[TissueAreas] = None  # L3-specific body composition


def _bone_lateral_spread(hu_image: np.ndarray, body_mask: np.ndarray,
                         bone_threshold: float = 150.0) -> Tuple[float, float, float]:
    """Compute bone features for a single slice.

    Returns:
        (total_bone_area, lateral_spread, bone_centroid_col)
        - lateral_spread: width (in pixels) of the bounding box of bone
        - total_bone_area: number of bone pixels
    """
    bone = (hu_image > bone_threshold) & body_mask
    bone_pixels = np.sum(bone)

    if bone_pixels < 10:
        return 0.0, 0.0, 0.0

    cols_with_bone = np.any(bone, axis=0)
    if not np.any(cols_with_bone):
        return float(bone_pixels), 0.0, 0.0

    bone_cols = np.where(cols_with_bone)[0]
    lateral_spread = float(bone_cols[-1] - bone_cols[0])

    rows_with_bone = np.any(bone, axis=1)
    bone_rows = np.where(rows_with_bone)[0]
    centroid_col = float(np.mean(bone_cols))

    return float(bone_pixels), lateral_spread, centroid_col


def _detect_iliac_crest_top(
    series_loader,
    thresholds: HUThresholds,
    sample_step: int = 3,
) -> Optional[int]:
    """Find the most cranial slice where the iliac crests are present.

    The iliac crests are wide lateral bone structures. Going from caudal
    (bottom) to cranial (top), bone lateral spread drops dramatically
    when we move above the iliac crests to the lumbar vertebrae alone.

    Returns the slice index of the top of the iliac crest (approximately L4-L5).
    """
    n = series_loader.num_slices
    spreads = np.zeros(n)

    # Sample slices for speed (bone features are smooth along z)
    sampled_indices = list(range(0, n, sample_step))

    for i in sampled_indices:
        hu_image, _ = series_loader.load_slice_hu(i)
        body_mask = (hu_image > thresholds.body_threshold)
        # Quick body mask — just threshold, no morphology needed for bone detection
        _, spread, _ = _bone_lateral_spread(hu_image, body_mask, thresholds.bone_threshold)
        spreads[i] = spread
        del hu_image

    # Interpolate unsampled slices
    for i in range(n):
        if i not in sampled_indices:
            # Find nearest sampled neighbors
            below = max(s for s in sampled_indices if s <= i)
            above = min(s for s in sampled_indices if s >= i)
            if below == above:
                spreads[i] = spreads[below]
            else:
                frac = (i - below) / (above - below)
                spreads[i] = spreads[below] + frac * (spreads[above] - spreads[below])

    # The iliac crest region has much wider bone than lumbar vertebrae alone.
    # Vertebral body alone: ~40-60 pixels wide
    # Iliac crest: ~150-300+ pixels wide
    #
    # Find the transition point where spread drops significantly going cranially.
    # We look for the cranial-most slice where spread is still "wide"
    # (above the vertebral-only baseline).

    # Compute a robust baseline from the middle third (likely lumbar/lower thoracic)
    mid_start = n // 3
    mid_end = 2 * n // 3
    mid_spreads = spreads[mid_start:mid_end]
    if len(mid_spreads) == 0:
        return None

    # The vertebral body baseline: typical spread for vertebrae alone
    # Use the 25th percentile of the middle section as "vertebral only" baseline
    vertebral_baseline = np.percentile(mid_spreads[mid_spreads > 0], 25) if np.any(mid_spreads > 0) else 0

    if vertebral_baseline == 0:
        return None

    # Iliac crest threshold: significantly wider than vertebral body alone
    # Typically 2-3x the vertebral body width
    iliac_threshold = vertebral_baseline * 1.8

    # Scan from the caudal end cranially, find where spread drops below threshold
    # The slices are sorted by slice_location (ascending), so the first slices
    # are the most caudal (or cranial depending on scan direction).
    # We need to figure out the direction.

    # Look at the caudal half for wide bone (pelvis)
    caudal_half = spreads[:n // 2]
    cranial_half = spreads[n // 2:]

    # Pelvis has wider bone. Determine which end has the pelvis.
    caudal_max = np.max(caudal_half) if len(caudal_half) > 0 else 0
    cranial_max = np.max(cranial_half) if len(cranial_half) > 0 else 0

    if caudal_max > cranial_max:
        # Pelvis is in the first half (ascending slice locations = caudal first)
        # Scan from caudal to cranial (low index to high index)
        # Find last slice (going cranially) where spread > threshold
        for i in range(n - 1, -1, -1):
            if spreads[i] > iliac_threshold:
                # This is still pelvis. The next cranial slice is the top.
                # But we want the transition, so scan forward from pelvis
                break

        # More robustly: find the cranial-most index where spread > threshold
        # within the caudal third of the scan
        pelvic_indices = np.where(spreads > iliac_threshold)[0]
        if len(pelvic_indices) == 0:
            return None
        return int(pelvic_indices[-1])  # Most cranial wide-bone slice
    else:
        # Pelvis is in the second half (descending, or scan is feet-first)
        pelvic_indices = np.where(spreads > iliac_threshold)[0]
        if len(pelvic_indices) == 0:
            return None
        return int(pelvic_indices[0])  # Most cranial wide-bone slice


def _detect_lowest_rib(
    series_loader,
    thresholds: HUThresholds,
    iliac_crest_index: int,
    sample_step: int = 3,
) -> Optional[int]:
    """Find the lowest rib-bearing slice (approximately T12/L1 junction).

    Ribs appear as small, laterally-placed bone fragments separate from
    the central vertebral body. Above T12, ribs are present; below L1
    they are absent.

    Searches only above the iliac crest to avoid confusion with pelvic bone.

    Returns the slice index of the lowest rib.
    """
    n = series_loader.num_slices

    # Determine scan direction
    locs = [series_loader._slice_locations[i] for i in range(n)]
    ascending = locs[0] is not None and locs[-1] is not None and locs[0] < locs[-1]

    if ascending:
        # Higher index = more cranial. Search from iliac crest upward.
        search_start = iliac_crest_index
        search_end = n
    else:
        search_start = 0
        search_end = iliac_crest_index

    for idx in range(search_start, search_end, sample_step):
        hu_image, _ = series_loader.load_slice_hu(idx)
        body_mask = (hu_image > thresholds.body_threshold)
        bone = (hu_image > thresholds.bone_threshold) & body_mask

        if np.sum(bone) < 10:
            del hu_image
            continue

        # Find the vertebral body: largest connected bone component (usually central/posterior)
        from skimage import measure as sk_measure
        labeled = sk_measure.label(bone)
        regions = sk_measure.regionprops(labeled)

        if not regions:
            del hu_image
            continue

        # Sort by area, largest = vertebral body
        regions.sort(key=lambda r: r.area, reverse=True)
        vertebral_body = regions[0]
        vb_centroid_col = vertebral_body.centroid[1]
        vb_bbox_width = vertebral_body.bbox[3] - vertebral_body.bbox[1]

        # Check for rib-like structures: small bone regions lateral to vertebral body
        has_ribs = False
        for region in regions[1:]:
            if region.area < 5:
                continue
            # Rib criteria:
            # 1. Laterally displaced from vertebral body centroid
            # 2. Relatively small compared to vertebral body
            # 3. Not too close to the vertebral body (not transverse process)
            region_centroid_col = region.centroid[1]
            lateral_dist = abs(region_centroid_col - vb_centroid_col)

            # Ribs are at least 1.5x the vertebral body width away from center
            if lateral_dist > vb_bbox_width * 0.8 and region.area < vertebral_body.area * 0.5:
                has_ribs = True
                break

        del hu_image

        if not has_ribs:
            continue

        # Found ribs at this level. Now find the most caudal rib-bearing slice
        # by refining the search.
        if ascending:
            # Search backward (caudally) from here to find exactly where ribs disappear
            for refine_idx in range(idx, search_start - 1, -1):
                hu_img, _ = series_loader.load_slice_hu(refine_idx)
                bm = (hu_img > thresholds.body_threshold)
                bn = (hu_img > thresholds.bone_threshold) & bm
                del hu_img

                if np.sum(bn) < 10:
                    continue

                lb = sk_measure.label(bn)
                rgs = sk_measure.regionprops(lb)
                if not rgs:
                    continue

                rgs.sort(key=lambda r: r.area, reverse=True)
                vb = rgs[0]
                vb_cc = vb.centroid[1]
                vb_bw = vb.bbox[3] - vb.bbox[1]

                found_rib = False
                for rg in rgs[1:]:
                    if rg.area < 5:
                        continue
                    ld = abs(rg.centroid[1] - vb_cc)
                    if ld > vb_bw * 0.8 and rg.area < vb.area * 0.5:
                        found_rib = True
                        break

                if not found_rib:
                    return refine_idx + 1  # One slice above = lowest rib
            return idx
        else:
            # Descending: search forward from here
            for refine_idx in range(idx, search_end):
                hu_img, _ = series_loader.load_slice_hu(refine_idx)
                bm = (hu_img > thresholds.body_threshold)
                bn = (hu_img > thresholds.bone_threshold) & bm
                del hu_img

                if np.sum(bn) < 10:
                    continue

                lb = sk_measure.label(bn)
                rgs = sk_measure.regionprops(lb)
                if not rgs:
                    continue

                rgs.sort(key=lambda r: r.area, reverse=True)
                vb = rgs[0]
                vb_cc = vb.centroid[1]
                vb_bw = vb.bbox[3] - vb.bbox[1]

                found_rib = False
                for rg in rgs[1:]:
                    if rg.area < 5:
                        continue
                    ld = abs(rg.centroid[1] - vb_cc)
                    if ld > vb_bw * 0.8 and rg.area < vb.area * 0.5:
                        found_rib = True
                        break

                if not found_rib:
                    return refine_idx - 1  # One slice above = lowest rib
            return idx

    return None


def _estimate_l3_from_landmarks(
    iliac_crest_index: int,
    lowest_rib_index: Optional[int],
    num_slices: int,
    ascending: bool,
) -> int:
    """Estimate the L3 slice index from anatomical landmarks.

    Between the iliac crest top (≈L4/L5) and the lowest rib (≈T12):
    - L4: just above iliac crest
    - L3: one vertebral body above L4
    - L2: two above
    - L1: three above
    - T12: at the lowest rib

    L3 is approximately 25% of the distance from iliac crest to T12
    (1 out of 4 inter-vertebral spaces: L4→L3→L2→L1→T12).
    """
    if lowest_rib_index is not None:
        if ascending:
            # Higher index = more cranial
            span = lowest_rib_index - iliac_crest_index
            l3_index = iliac_crest_index + int(span * 0.25)
        else:
            span = iliac_crest_index - lowest_rib_index
            l3_index = iliac_crest_index - int(span * 0.25)
    else:
        # No rib detected — estimate using typical vertebral body height.
        # Average L3 vertebral body height ≈ 30mm.
        # L3 center is approximately 1.5 vertebral heights above iliac crest.
        # With 1.25mm slices, that's about 36 slices.
        # Use a conservative estimate: 30 slices ≈ 37.5mm above iliac crest top.
        offset = 30  # slices
        if ascending:
            l3_index = iliac_crest_index + offset
        else:
            l3_index = iliac_crest_index - offset

    return max(0, min(l3_index, num_slices - 1))


def _refine_l3_by_sma_peak(
    series_loader,
    thresholds: HUThresholds,
    estimated_l3: int,
    search_radius_slices: int = 20,
) -> int:
    """Refine L3 estimate by finding the local SMA peak near the estimate.

    L3 typically has high SMA due to the prominent psoas, erector spinae,
    and abdominal wall muscles. The local SMA maximum near the anatomical
    estimate is likely the true L3 mid-vertebral level.
    """
    n = series_loader.num_slices
    start = max(0, estimated_l3 - search_radius_slices)
    end = min(n, estimated_l3 + search_radius_slices + 1)

    pixel_area_cm2 = (
        series_loader.pixel_spacing[0] * series_loader.pixel_spacing[1]
    ) / 100.0

    best_sma = -1.0
    best_idx = estimated_l3

    for i in range(start, end):
        hu_image, _ = series_loader.load_slice_hu(i)

        bmg = BodyMaskGenerator(hu_image, thresholds.body_threshold)
        body_mask = bmg.generate_mask()

        mcg = MuscleCompartmentGenerator(hu_image, body_mask, thresholds)
        mcg.generate_muscle_compartment(
            method='connectivity',
            sat_fraction=0.08,  # Standard L3 fractions
            muscle_fraction=0.12,
        )

        seg = TissueSegmenter(hu_image, body_mask, mcg, thresholds)
        masks = seg.get_all_masks()
        calc = AreaCalculator(masks, pixel_area_cm2)
        areas = calc.calculate_areas()

        if areas.sma > best_sma:
            best_sma = areas.sma
            best_idx = i

        del hu_image

    return best_idx


def detect_l3(
    series_loader,
    thresholds: Optional[HUThresholds] = None,
    progress_callback=None,
) -> L3DetectionResult:
    """Automatically detect the L3 vertebral level from a CT DICOM series.

    This is the main entry point. It combines multiple anatomical
    landmark detection methods and validates the result.

    Args:
        series_loader: DICOMSeriesLoader with the CT series
        thresholds: HU thresholds (defaults to standard values)
        progress_callback: Optional (current_step, total_steps) callback

    Returns:
        L3DetectionResult with slice index, location, confidence, and areas
    """
    if thresholds is None:
        thresholds = HUThresholds()

    n = series_loader.num_slices
    if n < 10:
        raise ValueError("Too few slices for L3 detection (need at least 10)")

    total_steps = 4
    step = 0

    # Step 1: Detect iliac crest
    if progress_callback:
        step += 1
        progress_callback(step, total_steps)

    iliac_crest_idx = _detect_iliac_crest_top(series_loader, thresholds, sample_step=3)

    # Determine scan direction
    loc0 = series_loader._slice_locations[0]
    loc_last = series_loader._slice_locations[-1]
    ascending = loc0 is not None and loc_last is not None and float(loc0) < float(loc_last)

    # Step 2: Detect lowest rib
    if progress_callback:
        step += 1
        progress_callback(step, total_steps)

    lowest_rib_idx = None
    if iliac_crest_idx is not None:
        lowest_rib_idx = _detect_lowest_rib(
            series_loader, thresholds, iliac_crest_idx, sample_step=3
        )

    # Step 3: Estimate L3 position
    if progress_callback:
        step += 1
        progress_callback(step, total_steps)

    confidence = 0.0
    method_parts = []

    if iliac_crest_idx is not None:
        estimated_l3 = _estimate_l3_from_landmarks(
            iliac_crest_idx, lowest_rib_idx, n, ascending
        )
        confidence = 0.7 if lowest_rib_idx is not None else 0.5
        method_parts.append("iliac crest landmark")
        if lowest_rib_idx is not None:
            method_parts.append("lowest rib landmark")
    else:
        # Fallback: use the peak SMA slice from the volumetric per-slice data.
        # This is less reliable but better than nothing.
        # Estimate L3 at approximately 60-65% of the scan from the top
        # (for a typical chest-to-pelvis scan).
        estimated_l3 = int(n * 0.65) if ascending else int(n * 0.35)
        confidence = 0.3
        method_parts.append("scan-position heuristic (no landmarks found)")

    # Step 4: Refine by finding local SMA peak near the estimate
    if progress_callback:
        step += 1
        progress_callback(step, total_steps)

    refined_l3 = _refine_l3_by_sma_peak(
        series_loader, thresholds, estimated_l3, search_radius_slices=15
    )

    if refined_l3 != estimated_l3:
        confidence = min(confidence + 0.15, 1.0)
        method_parts.append("SMA-peak refinement")

    # Analyze the detected L3 slice with standard L3 parameters
    pixel_area_cm2 = (
        series_loader.pixel_spacing[0] * series_loader.pixel_spacing[1]
    ) / 100.0

    hu_image, _ = series_loader.load_slice_hu(refined_l3)

    bmg = BodyMaskGenerator(hu_image, thresholds.body_threshold)
    body_mask = bmg.generate_mask()

    mcg = MuscleCompartmentGenerator(hu_image, body_mask, thresholds)
    mcg.generate_muscle_compartment(
        method='connectivity',
        sat_fraction=0.08,   # Standard L3 SAT fraction
        muscle_fraction=0.12,  # Standard L3 muscle band
    )

    seg = TissueSegmenter(hu_image, body_mask, mcg, thresholds)
    masks = seg.get_all_masks()
    calc = AreaCalculator(masks, pixel_area_cm2)
    l3_areas = calc.calculate_areas()

    del hu_image

    # Validate: check if SMA is in a reasonable range for L3
    # (very loose sanity check: >20 cm² and <300 cm²)
    if 20 < l3_areas.sma < 300:
        confidence = min(confidence + 0.1, 1.0)
    else:
        confidence = max(confidence - 0.2, 0.1)

    slice_location = series_loader._slice_locations[refined_l3]
    if slice_location is not None:
        slice_location = float(slice_location)

    return L3DetectionResult(
        l3_slice_index=refined_l3,
        l3_slice_location=slice_location,
        confidence=confidence,
        method=" + ".join(method_parts),
        iliac_crest_index=iliac_crest_idx,
        lowest_rib_index=lowest_rib_idx,
        l3_areas=l3_areas,
    )
