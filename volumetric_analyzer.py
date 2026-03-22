"""
Volumetric (3D) Body Composition Analyzer
==========================================
Processes an entire DICOM series (any number of slices) to compute
tissue VOLUMES (cm³) instead of single-slice areas (cm²).

MEMORY-EFFICIENT DESIGN:
- Only stores per-slice NUMBERS (areas), not pixel arrays
- Slice overlays are generated on-demand by re-reading the DICOM file
- DICOMSeriesLoader reads headers only (stop_before_pixels=True)
- Full pixel data is loaded one slice at a time during analysis

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import os
import gc
import numpy as np
import pydicom
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from l3_analyzer import (
    HUThresholds,
    TissueAreas,
    ClinicalRatios,
    DICOMLoader,
    BodyMaskGenerator,
    MuscleCompartmentGenerator,
    TissueSegmenter,
    AreaCalculator,
    Visualizer,
)
from l3_detector import detect_l3, L3DetectionResult


@dataclass
class TissueVolumes:
    """Tissue volumes in cm³ computed over the full DICOM series."""
    nama: float
    lama: float
    imat: float
    sma: float
    tama: float
    psoas: float = 0.0
    sat: float = 0.0


@dataclass
class VolumetricRatios:
    """Clinical ratios computed from volumetric data."""
    smv_bmi: float      # Skeletal Muscle Volume / BMI
    nama_vol_bmi: float  # NAMA volume / BMI
    lama_vol_bmi: float  # LAMA volume / BMI
    nama_tama_vol: float # NAMA volume / TAMA volume (muscle quality)


@dataclass
class SliceResult:
    """Per-slice analysis result — lightweight, stores only numbers."""
    slice_index: int
    slice_location: Optional[float]
    instance_number: Optional[int]
    areas: TissueAreas


class DICOMSeriesLoader:
    """Loads and sorts an entire DICOM series from multiple files.

    MEMORY-EFFICIENT: reads only DICOM headers during init (no pixel data).
    Pixel data is loaded on-demand via load_slice_hu().
    """

    def __init__(self, dicom_paths: List[str]):
        if not dicom_paths:
            raise ValueError("No DICOM files provided")

        self.sorted_paths: List[str] = []
        self.slice_thickness: float = 1.0
        self.pixel_spacing: Tuple[float, float] = (1.0, 1.0)
        self.num_slices: int = 0
        self._slice_locations: List[Optional[float]] = []
        self._instance_numbers: List[Optional[int]] = []
        self._series_metadata: Dict[str, Any] = {}

        self._load_and_sort(dicom_paths)

    def _load_and_sort(self, dicom_paths: List[str]) -> None:
        """Read headers only, sort by position, store paths."""
        entries = []
        for path in dicom_paths:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                # Must have basic image attributes
                if not hasattr(ds, 'Rows') or not hasattr(ds, 'Columns'):
                    continue
                loc = getattr(ds, 'SliceLocation', None)
                inst = getattr(ds, 'InstanceNumber', None)
                entries.append((path, ds, loc, inst))
            except Exception:
                continue

        if not entries:
            raise ValueError("No valid DICOM files found")

        # Sort by SliceLocation > InstanceNumber > filename
        def sort_key(item):
            _, _, loc, inst = item
            if loc is not None:
                return float(loc)
            if inst is not None:
                return int(inst)
            return 0

        entries.sort(key=sort_key)

        self.sorted_paths = [p for p, _, _, _ in entries]
        self._slice_locations = [loc for _, _, loc, _ in entries]
        self._instance_numbers = [inst for _, _, _, inst in entries]
        self.num_slices = len(entries)

        # Extract metadata from first header
        ds0 = entries[0][1]
        self.pixel_spacing = tuple(
            float(x) for x in getattr(ds0, 'PixelSpacing', [1.0, 1.0])
        )

        # Infer slice thickness
        sbs = getattr(ds0, 'SpacingBetweenSlices', None)
        st_ = getattr(ds0, 'SliceThickness', None)
        if sbs is not None:
            self.slice_thickness = float(sbs)
        elif st_ is not None:
            self.slice_thickness = float(st_)
        elif self.num_slices >= 2 and entries[0][2] is not None and entries[1][2] is not None:
            self.slice_thickness = abs(float(entries[1][2]) - float(entries[0][2]))
        else:
            self.slice_thickness = 1.0

        # Cache series-level metadata so we don't need to keep datasets
        self._series_metadata = {
            'patient_id': getattr(ds0, 'PatientID', 'Unknown'),
            'patient_name': str(getattr(ds0, 'PatientName', 'Unknown')),
            'study_date': getattr(ds0, 'StudyDate', 'Unknown'),
            'series_description': getattr(ds0, 'SeriesDescription', 'Unknown'),
            'num_slices': self.num_slices,
            'slice_thickness_mm': self.slice_thickness,
            'pixel_spacing': self.pixel_spacing,
            'rows': ds0.Rows,
            'columns': ds0.Columns,
        }
        # Datasets are NOT stored — headers are discarded here

    def get_series_metadata(self) -> Dict[str, Any]:
        return dict(self._series_metadata)

    def load_slice_hu(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a single slice's pixel data and convert to HU.

        This is the ONLY method that reads pixel data from disk.
        """
        ds = pydicom.dcmread(self.sorted_paths[index])
        raw = ds.pixel_array.astype(np.float64)

        rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))
        rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu_image = raw * rescale_slope + rescale_intercept

        meta = {
            'slice_location': self._slice_locations[index],
            'instance_number': self._instance_numbers[index],
            'slice_index': index,
        }
        return hu_image, meta


def _analyze_single_slice(
    hu_image: np.ndarray,
    thresholds: HUThresholds,
    pixel_area_cm2: float,
    sat_fraction: float = 0.08,
    muscle_fraction: float = 0.12,
) -> TissueAreas:
    """Run the full segmentation pipeline on one slice and return areas.

    All intermediate arrays (masks, body_mask, etc.) are discarded after
    this function returns, keeping memory usage constant.
    """
    bmg = BodyMaskGenerator(hu_image, thresholds.body_threshold)
    body_mask = bmg.generate_mask()

    mcg = MuscleCompartmentGenerator(hu_image, body_mask, thresholds)
    mcg.generate_muscle_compartment(
        method='connectivity',
        sat_fraction=sat_fraction,
        muscle_fraction=muscle_fraction,
    )

    seg = TissueSegmenter(hu_image, body_mask, mcg, thresholds)
    masks = seg.get_all_masks()

    calc = AreaCalculator(masks, pixel_area_cm2)
    return calc.calculate_areas()


class VolumetricAnalyzer:
    """Full 3D volumetric body composition analysis.

    MEMORY-EFFICIENT: processes one slice at a time, discards pixel data,
    stores only the numeric results (TissueAreas) per slice.
    """

    # Muscle band for multi-level volumetric analysis.
    # Must balance capturing deep muscles (psoas) vs. excluding organs.
    # 0.15 gives ~4-6 cm depth at typical body sizes, enough for
    # abdominal wall muscles + psoas without reaching visceral organs.
    VOL_SAT_FRACTION = 0.08    # Match L3 SAT zone depth
    VOL_MUSCLE_FRACTION = 0.15  # Moderate band: captures muscles, excludes organs

    def __init__(
        self,
        series_loader: DICOMSeriesLoader,
        thresholds: Optional[HUThresholds] = None,
    ):
        self.series = series_loader
        self.thresholds = thresholds or HUThresholds()

        self.slice_results: List[SliceResult] = []
        self._volumes: Optional[TissueVolumes] = None
        self._ratios: Optional[VolumetricRatios] = None

    def analyze(
        self,
        height_m: float,
        weight_kg: float,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run full volumetric analysis (memory-efficient).

        Processes one slice at a time. Only numeric areas are retained.
        """
        pixel_area_cm2 = (
            self.series.pixel_spacing[0] * self.series.pixel_spacing[1]
        ) / 100.0
        slice_thickness_cm = self.series.slice_thickness / 10.0

        self.slice_results = []

        for i in range(self.series.num_slices):
            # Load pixel data for this slice only
            hu_image, slice_meta = self.series.load_slice_hu(i)

            # Analyze with wider muscle band for multi-level coverage
            areas = _analyze_single_slice(
                hu_image, self.thresholds, pixel_area_cm2,
                sat_fraction=self.VOL_SAT_FRACTION,
                muscle_fraction=self.VOL_MUSCLE_FRACTION,
            )

            self.slice_results.append(SliceResult(
                slice_index=i,
                slice_location=slice_meta.get('slice_location'),
                instance_number=slice_meta.get('instance_number'),
                areas=areas,
            ))

            # Explicitly free the large arrays
            del hu_image
            if i % 20 == 0:
                gc.collect()

            if progress_callback:
                progress_callback(i + 1, self.series.num_slices)

        # Final GC after the loop
        gc.collect()

        # Integrate: volume = Σ (area_i × slice_thickness)
        self._volumes = TissueVolumes(
            nama=sum(s.areas.nama for s in self.slice_results) * slice_thickness_cm,
            lama=sum(s.areas.lama for s in self.slice_results) * slice_thickness_cm,
            imat=sum(s.areas.imat for s in self.slice_results) * slice_thickness_cm,
            sma=sum(s.areas.sma for s in self.slice_results) * slice_thickness_cm,
            tama=sum(s.areas.tama for s in self.slice_results) * slice_thickness_cm,
            psoas=sum(s.areas.psoas for s in self.slice_results) * slice_thickness_cm,
            sat=sum(s.areas.sat for s in self.slice_results) * slice_thickness_cm,
        )

        bmi = weight_kg / (height_m ** 2)
        self._ratios = VolumetricRatios(
            smv_bmi=self._volumes.sma / bmi if bmi > 0 else 0,
            nama_vol_bmi=self._volumes.nama / bmi if bmi > 0 else 0,
            lama_vol_bmi=self._volumes.lama / bmi if bmi > 0 else 0,
            nama_tama_vol=(
                self._volumes.nama / self._volumes.tama
                if self._volumes.tama > 0 else 0
            ),
        )

        # Automatic L3 detection
        self._l3_result: Optional[L3DetectionResult] = None
        try:
            self._l3_result = detect_l3(self.series, self.thresholds)
        except Exception:
            pass  # L3 detection is best-effort

        result = {
            'volumes': self._volumes,
            'ratios': self._ratios,
            'per_slice': self.slice_results,
            'metadata': self.series.get_series_metadata(),
            'bmi': bmi,
        }

        if self._l3_result is not None:
            result['l3_detection'] = self._l3_result

        return result

    def get_slice_overlay(
        self,
        slice_index: int,
        window_center: int = 40,
        window_width: int = 400,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Generate RGB overlay for a specific slice ON-DEMAND.

        Re-reads the DICOM file and re-runs segmentation for just this
        one slice. This uses memory only while the overlay is being
        generated, then frees it.
        """
        pixel_area_cm2 = (
            self.series.pixel_spacing[0] * self.series.pixel_spacing[1]
        ) / 100.0

        hu_image, _ = self.series.load_slice_hu(slice_index)

        bmg = BodyMaskGenerator(hu_image, self.thresholds.body_threshold)
        body_mask = bmg.generate_mask()

        mcg = MuscleCompartmentGenerator(hu_image, body_mask, self.thresholds)
        mcg.generate_muscle_compartment(
            method='connectivity',
            sat_fraction=self.VOL_SAT_FRACTION,
            muscle_fraction=self.VOL_MUSCLE_FRACTION,
        )

        seg = TissueSegmenter(hu_image, body_mask, mcg, self.thresholds)
        masks = seg.get_all_masks()

        viz = Visualizer(hu_image, masks, body_mask=body_mask)
        overlay = viz.generate_overlay(
            window_center=window_center,
            window_width=window_width,
            alpha=alpha,
        )

        return overlay

    def get_l3_overlay(
        self,
        window_center: int = 40,
        window_width: int = 400,
        alpha: float = 0.7,
    ) -> Optional[np.ndarray]:
        """Generate RGB overlay for the auto-detected L3 slice.

        Uses standard L3 parameters (8% SAT, 12% muscle band) for
        clinically validated segmentation.
        """
        if self._l3_result is None:
            return None

        idx = self._l3_result.l3_slice_index
        hu_image, _ = self.series.load_slice_hu(idx)

        bmg = BodyMaskGenerator(hu_image, self.thresholds.body_threshold)
        body_mask = bmg.generate_mask()

        mcg = MuscleCompartmentGenerator(hu_image, body_mask, self.thresholds)
        mcg.generate_muscle_compartment(
            method='connectivity',
            sat_fraction=0.08,   # Standard L3
            muscle_fraction=0.12,  # Standard L3
        )

        seg = TissueSegmenter(hu_image, body_mask, mcg, self.thresholds)
        masks = seg.get_all_masks()

        viz = Visualizer(hu_image, masks, body_mask=body_mask)
        overlay = viz.generate_overlay(
            window_center=window_center,
            window_width=window_width,
            alpha=alpha,
        )
        return overlay

    def get_results_dict(self) -> Dict[str, float]:
        """Flat dict of volumetric results for export."""
        if self._volumes is None or self._ratios is None:
            raise RuntimeError("Call analyze() first")
        d = {
            'NAMA_cm3': self._volumes.nama,
            'LAMA_cm3': self._volumes.lama,
            'IMAT_cm3': self._volumes.imat,
            'SMA_cm3': self._volumes.sma,
            'TAMA_cm3': self._volumes.tama,
            'Psoas_cm3': self._volumes.psoas,
            'SAT_cm3': self._volumes.sat,
            'SMV_BMI': self._ratios.smv_bmi,
            'NAMA_vol_BMI': self._ratios.nama_vol_bmi,
            'LAMA_vol_BMI': self._ratios.lama_vol_bmi,
            'NAMA_TAMA_vol': self._ratios.nama_tama_vol,
        }
        if self._l3_result is not None and self._l3_result.l3_areas is not None:
            a = self._l3_result.l3_areas
            d['L3_slice_index'] = self._l3_result.l3_slice_index
            d['L3_slice_location'] = self._l3_result.l3_slice_location
            d['L3_confidence'] = self._l3_result.confidence
            d['L3_NAMA_cm2'] = a.nama
            d['L3_LAMA_cm2'] = a.lama
            d['L3_IMAT_cm2'] = a.imat
            d['L3_Psoas_cm2'] = a.psoas
            d['L3_SMA_cm2'] = a.sma
            d['L3_NAMA_TAMA'] = a.nama / a.sma if a.sma > 0 else 0
        return d

    def get_per_slice_dataframe(self):
        """Return a pandas DataFrame with per-slice areas."""
        import pandas as pd
        rows = []
        for sr in self.slice_results:
            rows.append({
                'slice_index': sr.slice_index,
                'slice_location': sr.slice_location,
                'NAMA_cm2': sr.areas.nama,
                'LAMA_cm2': sr.areas.lama,
                'IMAT_cm2': sr.areas.imat,
                'Psoas_cm2': sr.areas.psoas,
                'SMA_cm2': sr.areas.sma,
                'SAT_cm2': sr.areas.sat,
            })
        return pd.DataFrame(rows)
