"""
Volumetric (3D) Body Composition Analyzer
==========================================
Processes an entire DICOM series (any number of slices) to compute
tissue VOLUMES (cm³) instead of single-slice areas (cm²).

Pipeline:
1. Load all DICOM files in a series
2. Sort by SliceLocation / InstanceNumber
3. Run per-slice tissue segmentation using the existing L3Analyzer pipeline
4. Sum slice areas × slice thickness → tissue volumes in cm³
5. Provide per-slice results for browsing and a volume summary

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import os
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


@dataclass
class TissueVolumes:
    """Tissue volumes in cm³ computed over the full DICOM series."""
    nama: float
    lama: float
    imat: float
    sma: float
    tama: float
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
    """Per-slice analysis result."""
    slice_index: int
    slice_location: Optional[float]
    instance_number: Optional[int]
    areas: TissueAreas
    hu_image: np.ndarray
    body_mask: np.ndarray
    masks: Dict[str, np.ndarray]
    pixel_area_cm2: float


class DICOMSeriesLoader:
    """Loads and sorts an entire DICOM series from multiple files.

    Handles any number of slices — from a handful to thousands.
    Sorts slices by SliceLocation (preferred) or InstanceNumber.
    """

    def __init__(self, dicom_paths: List[str]):
        """
        Args:
            dicom_paths: List of paths to DICOM files belonging to one series.
        """
        if not dicom_paths:
            raise ValueError("No DICOM files provided")

        self.dicom_paths = dicom_paths
        self.datasets: List[pydicom.Dataset] = []
        self.sorted_paths: List[str] = []
        self.slice_thickness: float = 1.0  # mm, will be inferred
        self.pixel_spacing: Tuple[float, float] = (1.0, 1.0)
        self.num_slices: int = 0

        self._load_and_sort()

    def _load_and_sort(self) -> None:
        """Load all DICOM files and sort by spatial position."""
        loaded = []
        for path in self.dicom_paths:
            try:
                ds = pydicom.dcmread(path)
                # Only include files that actually have pixel data
                if hasattr(ds, 'pixel_array'):
                    loaded.append((path, ds))
            except Exception:
                # Skip files that can't be read (non-DICOM, corrupted, etc.)
                continue

        if not loaded:
            raise ValueError("No valid DICOM files with pixel data found")

        # Sort by SliceLocation if available, else InstanceNumber, else filename
        def sort_key(item):
            _, ds = item
            loc = getattr(ds, 'SliceLocation', None)
            if loc is not None:
                return float(loc)
            inst = getattr(ds, 'InstanceNumber', None)
            if inst is not None:
                return int(inst)
            return 0

        loaded.sort(key=sort_key)

        self.sorted_paths = [p for p, _ in loaded]
        self.datasets = [ds for _, ds in loaded]
        self.num_slices = len(self.datasets)

        # Extract common metadata from the first slice
        ds0 = self.datasets[0]
        self.pixel_spacing = tuple(
            float(x) for x in getattr(ds0, 'PixelSpacing', [1.0, 1.0])
        )

        # Infer slice thickness: prefer SpacingBetweenSlices, then SliceThickness,
        # then compute from the actual SliceLocation difference between slices.
        sbs = getattr(ds0, 'SpacingBetweenSlices', None)
        st_ = getattr(ds0, 'SliceThickness', None)

        if sbs is not None:
            self.slice_thickness = float(sbs)
        elif st_ is not None:
            self.slice_thickness = float(st_)
        elif self.num_slices >= 2:
            loc0 = getattr(self.datasets[0], 'SliceLocation', None)
            loc1 = getattr(self.datasets[1], 'SliceLocation', None)
            if loc0 is not None and loc1 is not None:
                self.slice_thickness = abs(float(loc1) - float(loc0))
            else:
                self.slice_thickness = 1.0
        else:
            self.slice_thickness = 1.0

    def get_series_metadata(self) -> Dict[str, Any]:
        """Return metadata for the whole series."""
        ds = self.datasets[0]
        return {
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
            'num_slices': self.num_slices,
            'slice_thickness_mm': self.slice_thickness,
            'pixel_spacing': self.pixel_spacing,
            'rows': ds.Rows,
            'columns': ds.Columns,
        }

    def load_slice_hu(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a single slice's HU image and metadata.

        Args:
            index: Zero-based slice index (in sorted order)

        Returns:
            (hu_image, slice_metadata)
        """
        ds = self.datasets[index]
        raw = ds.pixel_array.astype(np.float64)

        rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
        rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
        hu_image = raw * rescale_slope + rescale_intercept

        meta = {
            'slice_location': getattr(ds, 'SliceLocation', None),
            'instance_number': getattr(ds, 'InstanceNumber', None),
            'slice_index': index,
        }
        return hu_image, meta


class VolumetricAnalyzer:
    """Full 3D volumetric body composition analysis.

    Processes every slice in a DICOM series, computes per-slice tissue
    areas, then integrates over the z-axis to produce tissue volumes (cm³).
    """

    def __init__(
        self,
        series_loader: DICOMSeriesLoader,
        thresholds: Optional[HUThresholds] = None,
    ):
        self.series = series_loader
        self.thresholds = thresholds or HUThresholds()

        # Per-slice results (populated by analyze())
        self.slice_results: List[SliceResult] = []

        # Summary results
        self._volumes: Optional[TissueVolumes] = None
        self._ratios: Optional[VolumetricRatios] = None

    def analyze(
        self,
        height_m: float,
        weight_kg: float,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run full volumetric analysis.

        Args:
            height_m: Patient height in metres.
            weight_kg: Patient weight in kg.
            progress_callback: Optional callable(current, total) for UI updates.

        Returns:
            Dict with 'volumes', 'ratios', 'per_slice', 'metadata', 'bmi'.
        """
        pixel_area_cm2 = (
            self.series.pixel_spacing[0] * self.series.pixel_spacing[1]
        ) / 100.0
        slice_thickness_cm = self.series.slice_thickness / 10.0  # mm → cm

        self.slice_results = []

        for i in range(self.series.num_slices):
            hu_image, slice_meta = self.series.load_slice_hu(i)

            # Body mask
            bmg = BodyMaskGenerator(hu_image, self.thresholds.body_threshold)
            body_mask = bmg.generate_mask()

            # Muscle compartment
            mcg = MuscleCompartmentGenerator(hu_image, body_mask, self.thresholds)
            mcg.generate_muscle_compartment(method='connectivity')

            # Tissue segmentation
            seg = TissueSegmenter(hu_image, body_mask, mcg, self.thresholds)
            masks = seg.get_all_masks()

            # Areas for this slice
            calc = AreaCalculator(masks, pixel_area_cm2)
            areas = calc.calculate_areas()

            self.slice_results.append(SliceResult(
                slice_index=i,
                slice_location=slice_meta.get('slice_location'),
                instance_number=slice_meta.get('instance_number'),
                areas=areas,
                hu_image=hu_image,
                body_mask=body_mask,
                masks=masks,
                pixel_area_cm2=pixel_area_cm2,
            ))

            if progress_callback:
                progress_callback(i + 1, self.series.num_slices)

        # Integrate: volume = Σ (area_i × slice_thickness)
        self._volumes = TissueVolumes(
            nama=sum(s.areas.nama for s in self.slice_results) * slice_thickness_cm,
            lama=sum(s.areas.lama for s in self.slice_results) * slice_thickness_cm,
            imat=sum(s.areas.imat for s in self.slice_results) * slice_thickness_cm,
            sma=sum(s.areas.sma for s in self.slice_results) * slice_thickness_cm,
            tama=sum(s.areas.tama for s in self.slice_results) * slice_thickness_cm,
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

        return {
            'volumes': self._volumes,
            'ratios': self._ratios,
            'per_slice': self.slice_results,
            'metadata': self.series.get_series_metadata(),
            'bmi': bmi,
        }

    def get_slice_overlay(
        self,
        slice_index: int,
        window_center: int = 40,
        window_width: int = 400,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Generate RGB overlay for a specific slice.

        Args:
            slice_index: Which slice to visualise (0-based).
            window_center / window_width: CT windowing params.
            alpha: Overlay opacity.

        Returns:
            RGB numpy array.
        """
        sr = self.slice_results[slice_index]
        viz = Visualizer(sr.hu_image, sr.masks, body_mask=sr.body_mask)
        return viz.generate_overlay(
            window_center=window_center,
            window_width=window_width,
            alpha=alpha,
        )

    def get_results_dict(self) -> Dict[str, float]:
        """Flat dict of volumetric results for export."""
        if self._volumes is None or self._ratios is None:
            raise RuntimeError("Call analyze() first")
        return {
            'NAMA_cm3': self._volumes.nama,
            'LAMA_cm3': self._volumes.lama,
            'IMAT_cm3': self._volumes.imat,
            'SMA_cm3': self._volumes.sma,
            'TAMA_cm3': self._volumes.tama,
            'SAT_cm3': self._volumes.sat,
            'SMV_BMI': self._ratios.smv_bmi,
            'NAMA_vol_BMI': self._ratios.nama_vol_bmi,
            'LAMA_vol_BMI': self._ratios.lama_vol_bmi,
            'NAMA_TAMA_vol': self._ratios.nama_tama_vol,
        }

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
                'SMA_cm2': sr.areas.sma,
                'SAT_cm2': sr.areas.sat,
            })
        return pd.DataFrame(rows)
