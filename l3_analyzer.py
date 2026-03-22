"""
L3 Body Composition Analyzer - Version 2.0
==========================================
A research-grade Python module for automated body composition analysis 
of CT images at the L3 vertebra level.

VERSION 1.1 IMPROVEMENTS:
- Added muscle compartment masking to properly separate subcutaneous fat from IMAT
- IMAT now only includes fat within the muscle compartment (between/within muscles)
- Subcutaneous fat is identified and excluded from IMAT calculation
- Added visceral fat identification (optional)

Clinical References for HU Ranges:
- Aubrey et al. (2014) Cancer Cachexia Study - NAMA/LAMA thresholds
- Prado et al. (2008) Sarcopenia Definition - SMA ranges
- Martin et al. (2013) Myosteatosis Assessment - IMAT definition
- Shen W et al. (2004) - Adipose tissue compartmentalization

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image
from skimage import morphology, measure
from skimage.segmentation import clear_border
from scipy import ndimage
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TissueType(Enum):
    """Enumeration of tissue types with their HU ranges.
    
    Clinical Reference:
        These HU thresholds are validated in sarcopenia research literature.
        - Aubrey J et al., Current Oncology, 2014
        - Prado CM et al., Lancet Oncol, 2008
    """
    NAMA = "Normal Attenuation Muscle Area"  # +30 to +150 HU
    LAMA = "Low Attenuation Muscle Area"     # -29 to +29 HU
    IMAT = "Intermuscular Adipose Tissue"    # -190 to -30 HU
    SMA = "Total Skeletal Muscle Area"       # -29 to +150 HU (NAMA + LAMA)
    SAT = "Subcutaneous Adipose Tissue"      # -190 to -30 HU (outer layer)
    VAT = "Visceral Adipose Tissue"          # -190 to -30 HU (around organs)


@dataclass
class HUThresholds:
    """Hounsfield Unit thresholds for tissue segmentation.
    
    Attributes:
        nama_min: Lower bound for NAMA (+30 HU)
        nama_max: Upper bound for NAMA (+150 HU)
        lama_min: Lower bound for LAMA (-29 HU)
        lama_max: Upper bound for LAMA (+29 HU)
        imat_min: Lower bound for IMAT (-190 HU)
        imat_max: Upper bound for IMAT (-30 HU)
        body_threshold: Threshold to separate body from air (-500 HU)
        bone_threshold: Threshold for bone tissue (+150 HU)
    
    Clinical Reference:
        Standard thresholds from Martin L et al., J Clin Oncol, 2013
    """
    nama_min: int = 30
    nama_max: int = 150
    lama_min: int = -29
    lama_max: int = 29
    imat_min: int = -190
    imat_max: int = -30
    body_threshold: int = -500
    bone_threshold: int = 150
    
    @property
    def sma_min(self) -> int:
        """SMA lower bound equals LAMA lower bound."""
        return self.lama_min
    
    @property
    def sma_max(self) -> int:
        """SMA upper bound equals NAMA upper bound."""
        return self.nama_max


@dataclass
class TissueAreas:
    """Container for calculated tissue areas in cm².

    Attributes:
        nama: Normal Attenuation Muscle Area (cm²)
        lama: Low Attenuation Muscle Area (cm²)
        imat: Intermuscular Adipose Tissue (cm²)
        sma: Total Skeletal Muscle Area (cm²)
        tama: Total Abdominal Muscle Area (cm²)
        psoas: Psoas Muscle Area (cm²) - bilateral total
        sat: Subcutaneous Adipose Tissue (cm²)
    """
    nama: float
    lama: float
    imat: float
    sma: float
    tama: float
    psoas: float = 0.0  # Psoas muscle area (included in SMA)
    sat: float = 0.0  # Subcutaneous fat (for reference)


@dataclass
class ClinicalRatios:
    """Clinical ratios for sarcopenia and myosteatosis assessment.
    
    Attributes:
        sma_bmi: SMA normalized by BMI (cm²/(kg/m²))
        nama_bmi: NAMA normalized by BMI (cm²/(kg/m²))
        lama_bmi: LAMA normalized by BMI (cm²/(kg/m²))
        nama_tama: NAMA/TAMA ratio (indicator of muscle quality)
    
    Clinical Significance:
        - SMA/BMI: Primary metric for sarcopenia diagnosis
        - NAMA/TAMA: Key indicator of muscle quality (myosteatosis)
          Low ratio indicates fat infiltration in muscle
    """
    sma_bmi: float
    nama_bmi: float
    lama_bmi: float
    nama_tama: float


@dataclass
class VisualizationColors:
    """RGB color definitions for tissue overlay visualization.
    
    Colors are defined as RGB tuples normalized to [0, 1] range.
    """
    nama: Tuple[float, float, float] = (1.0, 0.0, 0.0)    # Red
    lama: Tuple[float, float, float] = (0.0, 1.0, 1.0)    # Cyan
    imat: Tuple[float, float, float] = (1.0, 1.0, 0.0)    # Yellow
    psoas: Tuple[float, float, float] = (0.0, 1.0, 0.0)   # Green (psoas muscles)
    sat: Tuple[float, float, float] = (0.0, 0.5, 0.0)     # Dark Green (subcutaneous fat)
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Black


class DICOMLoader:
    """Handles DICOM file loading and Hounsfield Unit conversion.
    
    This class manages the critical step of converting raw DICOM pixel 
    values to standardized Hounsfield Units (HU) using the DICOM metadata
    (RescaleSlope and RescaleIntercept).
    
    The conversion formula is:
        HU = pixel_value * RescaleSlope + RescaleIntercept
    
    Attributes:
        dicom_data: The loaded pydicom dataset
        hu_image: The image array converted to Hounsfield Units
        pixel_spacing: Physical spacing between pixels in mm (row, column)
        pixel_area_cm2: Area of a single pixel in cm²
    """
    
    def __init__(self, dicom_path: str):
        """Initialize the DICOM loader with a file path.
        
        Args:
            dicom_path: Path to the DICOM file
            
        Raises:
            FileNotFoundError: If the DICOM file doesn't exist
            pydicom.errors.InvalidDicomError: If the file is not valid DICOM
        """
        self.dicom_path = dicom_path
        self.dicom_data: Optional[pydicom.Dataset] = None
        self.hu_image: Optional[np.ndarray] = None
        self.pixel_spacing: Optional[Tuple[float, float]] = None
        self.pixel_area_cm2: Optional[float] = None
        
        self._load_and_convert()
    
    def _load_and_convert(self) -> None:
        """Load DICOM file and convert pixel values to Hounsfield Units."""
        self.dicom_data = pydicom.dcmread(self.dicom_path)
        
        # Extract pixel array
        raw_image = self.dicom_data.pixel_array.astype(np.float64)
        
        # Apply modality LUT (RescaleSlope and RescaleIntercept)
        # This is the critical HU conversion step
        rescale_slope = getattr(self.dicom_data, 'RescaleSlope', 1.0)
        rescale_intercept = getattr(self.dicom_data, 'RescaleIntercept', 0.0)
        
        self.hu_image = raw_image * rescale_slope + rescale_intercept
        
        # Extract pixel spacing for area calculations
        # PixelSpacing is [row_spacing, column_spacing] in mm
        self.pixel_spacing = tuple(
            float(x) for x in getattr(self.dicom_data, 'PixelSpacing', [1.0, 1.0])
        )
        
        # Calculate pixel area in cm² (convert from mm²)
        self.pixel_area_cm2 = (self.pixel_spacing[0] * self.pixel_spacing[1]) / 100.0
    
    def get_metadata(self) -> Dict[str, Any]:
        """Extract relevant DICOM metadata.
        
        Returns:
            Dictionary containing patient and scan metadata
        """
        ds = self.dicom_data
        return {
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
            'slice_location': getattr(ds, 'SliceLocation', None),
            'slice_thickness': getattr(ds, 'SliceThickness', None),
            'pixel_spacing': self.pixel_spacing,
            'rows': ds.Rows,
            'columns': ds.Columns,
            'rescale_slope': getattr(ds, 'RescaleSlope', 1.0),
            'rescale_intercept': getattr(ds, 'RescaleIntercept', 0.0),
        }


class JPEGLoader:
    """Handles JPEG image loading with approximate HU mapping.

    Since JPEG images do not contain DICOM metadata, this loader maps
    grayscale pixel values (0-255) to an approximate Hounsfield Unit range
    based on user-provided window settings.

    Default mapping assumes a standard abdominal CT window:
        Window Center = 40 HU, Window Width = 400 HU
        => pixel 0 maps to -160 HU, pixel 255 maps to +240 HU

    Body isolation uses Otsu's thresholding on raw pixel values to
    automatically separate the CT body region from the dark background
    (important for photos of monitor screens).

    Attributes:
        hu_image: The image array mapped to approximate Hounsfield Units
        pixel_spacing: Physical spacing between pixels in mm (row, column)
        pixel_area_cm2: Area of a single pixel in cm²
        body_threshold_hu: Auto-computed HU threshold for body/background separation
    """

    def __init__(self, jpeg_path: str,
                 pixel_spacing: Tuple[float, float] = (1.0, 1.0),
                 hu_min: float = -160.0,
                 hu_max: float = 240.0):
        """Initialize the JPEG loader.

        Args:
            jpeg_path: Path to the JPEG image file
            pixel_spacing: Physical pixel spacing in mm (row, col).
                           Default is 1.0mm x 1.0mm.
            hu_min: HU value corresponding to pixel value 0
            hu_max: HU value corresponding to pixel value 255
        """
        self.jpeg_path = jpeg_path
        self.pixel_spacing = pixel_spacing
        self.pixel_area_cm2 = (pixel_spacing[0] * pixel_spacing[1]) / 100.0
        self.hu_image: Optional[np.ndarray] = None
        self.body_threshold_hu: Optional[float] = None
        self.precomputed_body_mask: Optional[np.ndarray] = None
        self._hu_min = hu_min
        self._hu_max = hu_max

        self._load_and_convert()

    def _load_and_convert(self) -> None:
        """Load JPEG file and map pixel values to approximate HU range.

        Handles real-world JPEG images including photos of monitor screens:
        1. Convert to grayscale and denoise
        2. Compute robust body mask directly from pixel intensities
        3. Map pixel values to HU range
        """
        from skimage.filters import median
        from skimage.morphology import disk

        img = Image.open(self.jpeg_path).convert('L')  # grayscale
        raw = np.array(img, dtype=np.float64)

        # Denoise with median filter (handles photo noise from monitor captures)
        raw_uint8 = raw.astype(np.uint8)
        denoised = median(raw_uint8, disk(3))

        # Linear mapping: pixel 0 -> hu_min, pixel 255 -> hu_max
        self.hu_image = self._hu_min + (denoised.astype(np.float64) / 255.0) * (self._hu_max - self._hu_min)

        # Compute body mask directly from raw pixels (bypasses HU thresholding)
        self.precomputed_body_mask = self._compute_body_mask(denoised)

        # Set a body_threshold_hu for compatibility (won't actually be used)
        self.body_threshold_hu = self._hu_min + 0.15 * (self._hu_max - self._hu_min)

    def _compute_body_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Compute body mask directly from pixel intensities.

        Uses a dual-threshold + convex hull approach for photos of CT monitors:
        1. Otsu threshold to find definite bright tissue (muscle/bone seeds)
        2. Keep largest seed cluster (the body's bright tissue)
        3. Convex hull of seeds captures the overall body shape
        4. Low threshold finds all potential body pixels (including fat)
        5. Expand hull slightly and intersect with low-threshold mask
        6. Fill holes and morphological cleanup

        This correctly handles photos where >80% of the image is dark
        monitor background, which defeats percentile-based approaches.
        """
        from skimage.filters import threshold_otsu
        from skimage.morphology import convex_hull_image

        # Step 1: Otsu to find definite bright body tissue (muscle, bone, organs)
        try:
            otsu_thresh = threshold_otsu(pixels.astype(np.uint8))
        except ValueError:
            otsu_thresh = 50

        seeds = pixels > otsu_thresh

        # Step 2: Keep only the largest bright cluster (the body)
        labeled = measure.label(seeds)
        if labeled.max() > 0:
            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            body_seeds = (labeled == largest.label)
        else:
            body_seeds = seeds

        # Step 3: Convex hull of seeds - captures overall body outline
        hull = convex_hull_image(body_seeds)

        # Step 4: Low threshold to find all potential body pixels (fat, tissue)
        low_thresh = max(otsu_thresh * 0.35, 25)
        potential_body = pixels > low_thresh

        # Step 5: Expand hull slightly to catch boundary fat, then intersect
        hull_expanded = morphology.dilation(hull, morphology.disk(15))
        body_mask = (hull_expanded & potential_body) | hull

        # Step 6: Cleanup
        body_mask = morphology.closing(body_mask, morphology.disk(3))
        body_mask = ndimage.binary_fill_holes(body_mask)

        # Keep largest component only
        labeled2 = measure.label(body_mask)
        if labeled2.max() > 0:
            regions2 = measure.regionprops(labeled2)
            largest2 = max(regions2, key=lambda r: r.area)
            body_mask = (labeled2 == largest2.label)

        body_mask = ndimage.binary_fill_holes(body_mask)

        return body_mask.astype(bool)

    def get_metadata(self) -> Dict[str, Any]:
        """Return placeholder metadata for JPEG images.

        Returns:
            Dictionary containing available metadata
        """
        h, w = self.hu_image.shape
        return {
            'patient_id': 'N/A (JPEG)',
            'patient_name': 'N/A (JPEG)',
            'study_date': 'N/A',
            'series_description': 'JPEG Import',
            'slice_location': None,
            'slice_thickness': None,
            'pixel_spacing': self.pixel_spacing,
            'rows': h,
            'columns': w,
            'rescale_slope': 1.0,
            'rescale_intercept': 0.0,
        }


class BodyMaskGenerator:
    """Generates body masks to exclude CT bed and external artifacts.
    
    This class implements "Engineering-Led Cleaning" using morphological
    operations and Connected Component Analysis (CCA) to automatically
    isolate the patient's body from:
    - CT table/bed
    - External cables and artifacts
    - Air surrounding the patient
    
    The algorithm:
    1. Threshold to separate body from air (> -500 HU)
    2. Apply morphological closing to fill small holes
    3. Use connected component analysis to identify largest object (body)
    4. Remove small objects (noise, artifacts)
    5. Clear border-connected components (CT bed often touches edges)
    
    Attributes:
        hu_image: Input image in Hounsfield Units
        body_threshold: HU threshold for body/air separation
    """
    
    def __init__(self, hu_image: np.ndarray, body_threshold: int = -500):
        """Initialize the body mask generator.
        
        Args:
            hu_image: 2D numpy array of HU values
            body_threshold: HU value to separate body from air
        """
        self.hu_image = hu_image
        self.body_threshold = body_threshold
        self._body_mask: Optional[np.ndarray] = None
    
    def generate_mask(self, 
                      min_body_size: int = 5000,
                      closing_radius: int = 5,
                      remove_border_objects: bool = True) -> np.ndarray:
        """Generate a binary mask of the patient's body.
        
        Args:
            min_body_size: Minimum pixel count for valid body regions
            closing_radius: Radius for morphological closing operation
            remove_border_objects: Whether to remove objects touching image border
            
        Returns:
            Binary numpy array where True indicates body pixels
        """
        # Step 1: Initial thresholding - separate body from air
        initial_mask = self.hu_image > self.body_threshold
        
        # Step 2: Morphological closing to fill small holes and gaps
        # This connects nearby regions and smooths the boundary
        selem = morphology.disk(closing_radius)
        closed_mask = morphology.closing(initial_mask, selem)
        
        # Step 3: Fill holes in the body mask
        # Important for including air-filled structures inside the body
        filled_mask = ndimage.binary_fill_holes(closed_mask)
        
        # Step 4: Remove small objects (noise, artifacts)
        cleaned_mask = morphology.remove_small_objects(
            filled_mask,
            max_size=min_body_size
        )
        
        # Step 5: Connected Component Analysis - keep largest component
        labeled_mask = measure.label(cleaned_mask)
        if labeled_mask.max() > 0:
            # Find the largest connected component (the body)
            regions = measure.regionprops(labeled_mask)
            largest_region = max(regions, key=lambda r: r.area)
            body_mask = labeled_mask == largest_region.label
        else:
            body_mask = cleaned_mask
        
        # Step 6: Optionally remove border-connected objects (CT bed)
        if remove_border_objects:
            # Clear objects touching the border, but preserve the main body
            # by checking if most of the body would be removed
            test_cleared = clear_border(body_mask)
            if np.sum(test_cleared) > 0.5 * np.sum(body_mask):
                body_mask = test_cleared
        
        # Step 7: Final morphological cleanup
        # Apply erosion followed by dilation to smooth edges
        body_mask = morphology.opening(body_mask, morphology.disk(2))
        body_mask = morphology.closing(body_mask, morphology.disk(2))
        
        self._body_mask = body_mask.astype(bool)
        return self._body_mask
    
    @property
    def body_mask(self) -> np.ndarray:
        """Get the generated body mask."""
        if self._body_mask is None:
            return self.generate_mask()
        return self._body_mask


class MuscleCompartmentGenerator:
    """
    NEW CLASS: Generates a muscle compartment mask that separates
    subcutaneous adipose tissue (SAT) from the muscle region.
    
    This is CRITICAL for accurate IMAT measurement.
    
    In clinical body composition analysis at L3:
    - Subcutaneous fat is the outer layer of fat just under the skin
    - IMAT (Intermuscular Adipose Tissue) is fat WITHIN the muscle compartment
    - These must be separated for accurate clinical metrics
    
    VERSION 2.1 IMPROVEMENT:
    Uses CONNECTIVITY-BASED separation instead of pure morphological approach.
    Fat regions that connect to the body outline are classified as subcutaneous.
    Fat regions that are isolated within the body are classified as IMAT.
    
    Clinical Reference:
        Shen W et al. Am J Clin Nutr 2004 - Adipose tissue compartmentalization
        Mourtzakis M et al. Appl Physiol Nutr Metab 2008 - L3 analysis protocol
    """
    
    def __init__(self, 
                 hu_image: np.ndarray, 
                 body_mask: np.ndarray,
                 thresholds: HUThresholds):
        """Initialize the muscle compartment generator.
        
        Args:
            hu_image: 2D numpy array of HU values
            body_mask: Binary mask of the patient's body
            thresholds: HU thresholds for tissue classification
        """
        self.hu_image = hu_image
        self.body_mask = body_mask
        self.thresholds = thresholds
        
        self._muscle_compartment_mask: Optional[np.ndarray] = None
        self._subcutaneous_fat_mask: Optional[np.ndarray] = None
        self._imat_mask: Optional[np.ndarray] = None
        self._psoas_zone_mask: Optional[np.ndarray] = None
    
    def generate_muscle_compartment(self,
                                    boundary_dilation: int = 5,
                                    method: str = 'connectivity',
                                    sat_fraction: float = 0.08,
                                    muscle_fraction: float = 0.12) -> np.ndarray:
        """
        Generate a mask defining the muscle compartment boundary.
        
        The muscle compartment includes:
        - All skeletal muscle tissue (NAMA + LAMA)
        - Intermuscular fat (IMAT) between muscle groups
        - Excludes subcutaneous fat layer
        
        Args:
            boundary_dilation: Pixels to dilate body outline for connectivity check
            method: 'connectivity' (recommended) or 'morphological'
            
        Returns:
            Binary mask of the muscle compartment
        """
        
        if method == 'connectivity':
            return self._generate_connectivity_based(
                boundary_dilation, sat_fraction, muscle_fraction
            )
        elif method == 'morphological':
            return self._generate_morphological()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_connectivity_based(self, boundary_dilation: int = 5,
                                      sat_fraction: float = 0.08,
                                      muscle_fraction: float = 0.12) -> np.ndarray:
        """
        CONNECTIVITY-BASED approach to separate tissues into compartments.

        Produces three compartments:
        - Subcutaneous fat (SAT): fat connected to the body outline
        - Muscle compartment: peripheral muscle band + psoas zone near vertebra
        - Visceral cavity: organs and visceral fat in the center (EXCLUDED)

        Strategy:
        1. Identify body outline and classify fat as SAT vs interior fat
        2. Define peripheral muscle band (abdominal wall muscles)
        3. Detect vertebral body and add psoas zone around it
        4. IMAT = only SMALL fat pockets within the muscle compartment

        Clinical Reference:
            Shen W et al. Am J Clin Nutr 2004
            Mourtzakis M et al. Appl Physiol Nutr Metab 2008
        """

        from scipy.ndimage import distance_transform_edt

        # Step 1: Get the body outline (outer boundary)
        body_outline = self.body_mask & ~morphology.erosion(
            self.body_mask, morphology.disk(3)
        )

        # Dilate the outline to create a "near-boundary" zone
        near_boundary = morphology.dilation(
            body_outline, morphology.disk(boundary_dilation)
        )

        # Step 2: Identify all fat tissue within body
        all_fat = (
            (self.hu_image >= self.thresholds.imat_min) &
            (self.hu_image <= self.thresholds.imat_max) &
            self.body_mask
        )

        # Step 3: Label connected fat regions and classify SAT vs interior
        labeled_fat = measure.label(all_fat)
        subcutaneous_mask = np.zeros_like(all_fat)
        interior_fat_mask = np.zeros_like(all_fat)

        for region in measure.regionprops(labeled_fat):
            region_mask = labeled_fat == region.label
            dilated_region = morphology.dilation(region_mask, morphology.disk(3))

            if np.any(dilated_region & near_boundary):
                subcutaneous_mask |= region_mask
            else:
                interior_fat_mask |= region_mask

        # Step 4: Define PERIPHERAL muscle band (abdominal wall muscles)
        dist_from_boundary = distance_transform_edt(self.body_mask)

        body_rows = np.any(self.body_mask, axis=1)
        body_height = np.sum(body_rows)
        body_cols = np.any(self.body_mask, axis=0)
        body_width = np.sum(body_cols)
        body_minor = min(body_height, body_width)

        sat_depth = max(15, int(body_minor * sat_fraction))
        muscle_depth = max(20, int(body_minor * muscle_fraction))
        muscle_end = sat_depth + muscle_depth

        # Geometric SAT zone
        geometric_sat = self.body_mask & (dist_from_boundary <= sat_depth)
        subcutaneous_mask = subcutaneous_mask | geometric_sat

        # Peripheral muscle band (abdominal wall)
        muscle_band = self.body_mask & (
            (dist_from_boundary > sat_depth) &
            (dist_from_boundary <= muscle_end)
        )

        # Step 5: Detect vertebral body and add BILATERAL PSOAS ZONES
        #
        # At L3, the psoas major sits bilaterally adjacent to the vertebral
        # body and transverse processes. It is DEEP (beyond the peripheral
        # muscle band) and must be captured separately.
        #
        # Strategy:
        # - Find vertebral bone (HU > 200)
        # - Locate the vertebral body (largest bone cluster)
        # - Create TWO elliptical search zones: one LEFT, one RIGHT
        #   of the vertebra (psoas is lateral, not centered)
        # - The psoas zone is where we LOOK for muscle — actual psoas
        #   pixels are identified by SMA HU range within this zone
        bone_mask = (self.hu_image > 200) & self.body_mask
        bone_mask = morphology.remove_small_objects(bone_mask, max_size=100)

        psoas_zone = np.zeros_like(self.body_mask)
        if np.any(bone_mask):
            labeled_bone = measure.label(bone_mask)
            bone_regions = measure.regionprops(labeled_bone)
            if bone_regions:
                # Find the largest bone region (vertebral body)
                vertebra = max(bone_regions, key=lambda r: r.area)
                vert_centroid_r, vert_centroid_c = vertebra.centroid

                # Psoas zone parameters — sized relative to body
                # The psoas is an oval muscle ~3-5 cm long axis at L3
                psoas_r_vert = max(25, int(body_minor * 0.08))   # vertical radius
                psoas_r_lat = max(20, int(body_minor * 0.06))    # lateral radius
                # Lateral offset: psoas center is ~2-3 cm lateral to vertebra center
                lateral_offset = max(20, int(body_minor * 0.07))
                # Slight anterior offset: psoas is anterolateral to vertebral body
                anterior_offset = max(5, int(body_minor * 0.02))

                rr, cc = np.ogrid[:self.hu_image.shape[0], :self.hu_image.shape[1]]

                # Left psoas (higher column index = patient's left in standard view)
                left_center_r = vert_centroid_r - anterior_offset
                left_center_c = vert_centroid_c + lateral_offset
                left_ellipse = (
                    ((rr - left_center_r) / psoas_r_vert) ** 2 +
                    ((cc - left_center_c) / psoas_r_lat) ** 2
                ) <= 1.0

                # Right psoas (lower column index)
                right_center_r = vert_centroid_r - anterior_offset
                right_center_c = vert_centroid_c - lateral_offset
                right_ellipse = (
                    ((rr - right_center_r) / psoas_r_vert) ** 2 +
                    ((cc - right_center_c) / psoas_r_lat) ** 2
                ) <= 1.0

                psoas_zone = (left_ellipse | right_ellipse) & self.body_mask
                # Exclude vertebral bone itself, SAT zone, and peripheral band
                # (muscle already captured in peripheral band shouldn't be
                # double-counted as psoas)
                psoas_zone = psoas_zone & ~bone_mask & ~subcutaneous_mask

        # Step 6: Combined muscle compartment = peripheral band + psoas zone
        combined_valid_zone = muscle_band | psoas_zone

        # Only include muscle-density tissue in the compartment
        muscle_tissue = (
            (self.hu_image >= self.thresholds.sma_min) &
            (self.hu_image <= self.thresholds.sma_max) &
            combined_valid_zone
        )
        bone_tissue = (
            (self.hu_image > self.thresholds.bone_threshold) &
            muscle_band  # bone only in peripheral band, not psoas zone
        )

        structural = muscle_tissue | bone_tissue
        structural = morphology.remove_small_objects(structural, max_size=50)

        # Dilate slightly to include small IMAT pockets between muscle groups
        muscle_compartment = morphology.dilation(structural, morphology.disk(3))
        muscle_compartment = muscle_compartment & combined_valid_zone

        # Step 7: IMAT = small fat pockets WITHIN the muscle compartment only
        imat_mask = np.zeros_like(all_fat)
        interior_in_compartment = interior_fat_mask & combined_valid_zone

        labeled_interior = measure.label(interior_in_compartment)
        max_imat_size = 2000
        for region in measure.regionprops(labeled_interior):
            if region.area <= max_imat_size:
                imat_mask |= (labeled_interior == region.label)

        self._muscle_compartment_mask = muscle_compartment.astype(bool)
        self._subcutaneous_fat_mask = subcutaneous_mask.astype(bool)
        self._imat_mask = imat_mask.astype(bool)
        self._psoas_zone_mask = psoas_zone.astype(bool)

        return self._muscle_compartment_mask
    
    def _generate_morphological(self, dilation_radius: int = 3) -> np.ndarray:
        """
        Morphological approach to muscle compartment identification.
        
        Strategy:
        1. Identify all muscle tissue (SMA: -29 to +150 HU)
        2. Identify bone (> 150 HU) - vertebra is a landmark
        3. Combine muscle + bone to get "non-fat-soft-tissue"
        4. Dilate this to create a connected muscle compartment
        5. Fill holes to include IMAT between muscles
        
        This approach works because:
        - Muscles are distributed around the periphery of the abdominal wall
        - The muscle fascia connects adjacent muscles
        - Fat between these muscles (IMAT) is enclosed by this fascia
        """
        
        # Step 1: Identify muscle tissue (SMA range)
        muscle_mask = (
            (self.hu_image >= self.thresholds.sma_min) & 
            (self.hu_image <= self.thresholds.sma_max) &
            self.body_mask
        )
        
        # Step 2: Identify bone/dense tissue (vertebra, etc.)
        bone_mask = (
            (self.hu_image > self.thresholds.bone_threshold) &
            self.body_mask
        )
        
        # Step 3: Combine muscle and bone as "structural tissue"
        structural_tissue = muscle_mask | bone_mask
        
        # Step 4: Clean up small isolated regions
        structural_tissue = morphology.remove_small_objects(
            structural_tissue, max_size=100
        )
        
        # Step 5: Dilate to connect nearby muscle groups
        dilated_structure = morphology.dilation(
            structural_tissue, 
            morphology.disk(dilation_radius)
        )
        
        # Step 6: Keep only within body mask
        dilated_structure = dilated_structure & self.body_mask
        
        # Step 7: Fill holes to include IMAT pockets between muscles
        muscle_compartment = ndimage.binary_fill_holes(dilated_structure)
        
        # Step 8: Apply closing to smooth the boundary
        muscle_compartment = morphology.closing(
            muscle_compartment, 
            morphology.disk(2)
        )
        
        # Step 9: Ensure we stay within body mask
        muscle_compartment = muscle_compartment & self.body_mask
        
        self._muscle_compartment_mask = muscle_compartment.astype(bool)
        return self._muscle_compartment_mask
    
    def get_subcutaneous_fat_mask(self) -> np.ndarray:
        """
        Get the subcutaneous adipose tissue (SAT) mask.
        
        SAT = Fat connected to the body outline (outer fat layer)
        
        Returns:
            Binary mask of subcutaneous fat
        """
        if self._subcutaneous_fat_mask is None:
            # Run connectivity-based analysis if not already done
            self.generate_muscle_compartment(method='connectivity')
        
        return self._subcutaneous_fat_mask
    
    def get_imat_mask(self) -> np.ndarray:
        """
        Get the true IMAT mask (fat NOT connected to body outline).
        
        IMAT = Fat isolated within the muscle compartment
        
        Returns:
            Binary mask of intermuscular adipose tissue
        """
        if self._imat_mask is None:
            self.generate_muscle_compartment(method='connectivity')
        
        return self._imat_mask
    
    @property
    def muscle_compartment_mask(self) -> np.ndarray:
        """Get the muscle compartment mask."""
        if self._muscle_compartment_mask is None:
            self.generate_muscle_compartment()
        return self._muscle_compartment_mask
    
    @property
    def subcutaneous_fat_mask(self) -> np.ndarray:
        """Get the subcutaneous fat mask."""
        return self.get_subcutaneous_fat_mask()
    
    @property
    def imat_mask(self) -> np.ndarray:
        """Get the IMAT mask."""
        return self.get_imat_mask()

    @property
    def psoas_zone_mask(self) -> np.ndarray:
        """Get the psoas search zone mask (bilateral, around vertebra)."""
        if self._psoas_zone_mask is None:
            self.generate_muscle_compartment(method='connectivity')
        return self._psoas_zone_mask


class TissueSegmenter:
    """Segments tissues based on Hounsfield Unit thresholds.
    
    VERSION 2.1: Now uses connectivity-based IMAT calculation for accurate
    separation of subcutaneous fat from intermuscular fat.
    
    This class performs radiodensity-based muscle partitioning by
    binning pixels into clinically validated HU ranges:
    
    - NAMA (Normal Attenuation Muscle Area): +30 to +150 HU
      Represents healthy muscle with minimal fat infiltration
      
    - LAMA (Low Attenuation Muscle Area): -29 to +29 HU
      Represents muscle with significant lipid content (myosteatosis)
      
    - IMAT (Intermuscular Adipose Tissue): -190 to -30 HU
      Fat tissue located WITHIN the muscle compartment (between muscle groups)
      NOT including subcutaneous fat - uses connectivity-based separation
      
    - SMA (Total Skeletal Muscle Area): -29 to +150 HU
      Sum of NAMA and LAMA
    
    Clinical Reference:
        Mourtzakis M et al., Appl Physiol Nutr Metab, 2008
        Prado CM et al., Lancet Oncol, 2008
    
    Attributes:
        hu_image: Input image in Hounsfield Units
        body_mask: Binary mask defining valid analysis region
        muscle_compartment_generator: Generator for muscle compartment masks
        thresholds: HUThresholds object defining tissue boundaries
    """
    
    def __init__(self, 
                 hu_image: np.ndarray, 
                 body_mask: np.ndarray,
                 muscle_compartment_generator: MuscleCompartmentGenerator,
                 thresholds: Optional[HUThresholds] = None):
        """Initialize the tissue segmenter.
        
        Args:
            hu_image: 2D numpy array of HU values
            body_mask: Binary mask where True indicates valid body pixels
            muscle_compartment_generator: Generator that provides IMAT/SAT separation
            thresholds: Custom HU thresholds (uses defaults if None)
        """
        self.hu_image = hu_image
        self.body_mask = body_mask
        self.muscle_compartment_generator = muscle_compartment_generator
        self.thresholds = thresholds or HUThresholds()
        
        # Tissue masks (computed on demand)
        self._nama_mask: Optional[np.ndarray] = None
        self._lama_mask: Optional[np.ndarray] = None
        self._imat_mask: Optional[np.ndarray] = None
        self._sma_mask: Optional[np.ndarray] = None
        self._sat_mask: Optional[np.ndarray] = None
        self._psoas_mask: Optional[np.ndarray] = None
    
    def _create_tissue_mask(self, 
                            hu_min: int, 
                            hu_max: int,
                            region_mask: np.ndarray,
                            min_region_size: int = 50) -> np.ndarray:
        """Create a binary mask for pixels within specified HU range.
        
        Args:
            hu_min: Minimum HU value (inclusive)
            hu_max: Maximum HU value (inclusive)
            region_mask: Mask defining the valid region for this tissue
            min_region_size: Minimum pixel count for valid regions
            
        Returns:
            Binary numpy array of tissue mask
        """
        # Apply HU thresholding
        mask = (self.hu_image >= hu_min) & (self.hu_image <= hu_max)
        
        # Restrict to specified region
        mask = mask & region_mask
        
        # Remove small isolated regions (noise)
        if min_region_size > 0:
            mask = morphology.remove_small_objects(mask, max_size=min_region_size)
        
        return mask.astype(bool)
    
    def segment_sma(self) -> np.ndarray:
        """Segment Total Skeletal Muscle Area (-29 to +150 HU).

        Uses iterative region-growing from NAMA seeds:
        1. NAMA (30-150 HU) = definite muscle tissue (seed regions)
        2. LAMA (-29 to +29 HU) = grown outward from NAMA pixel-by-pixel
           using iterative dilation (flood-fill within muscle compartment)

        This avoids the connected-component flaw where an entire blob of
        non-muscle tissue gets included because one pixel touches NAMA.
        Instead, only LAMA pixels that are spatially contiguous with NAMA
        — reachable step-by-step through adjacent LAMA pixels — are counted.

        Clinical Reference:
            Mourtzakis M et al., Appl Physiol Nutr Metab, 2008
            Prado CM et al., Lancet Oncol, 2008

        Returns:
            Binary mask of SMA pixels
        """
        if self._sma_mask is None:
            muscle_region = self.muscle_compartment_generator.muscle_compartment_mask
            bone_exclusion = self.hu_image <= self.thresholds.bone_threshold
            valid_region = muscle_region & bone_exclusion

            # Step 1: Find NAMA (definite muscle) within muscle compartment
            nama_candidates = (
                (self.hu_image >= self.thresholds.nama_min) &
                (self.hu_image <= self.thresholds.nama_max) &
                valid_region
            )
            # Remove tiny noise regions
            nama_candidates = morphology.remove_small_objects(
                nama_candidates, max_size=50
            )

            # Step 2: Find LAMA candidates within muscle compartment
            lama_candidates = (
                (self.hu_image >= self.thresholds.lama_min) &
                (self.hu_image <= self.thresholds.lama_max) &
                valid_region
            )

            # Step 3: Iterative region-growing from NAMA into LAMA.
            # Start with NAMA as seeds. On each iteration, dilate by 1px
            # and pick up adjacent LAMA pixels. Repeat until no new LAMA
            # pixels are added. This is a flood-fill that stops at non-LAMA
            # boundaries, so organ tissue that merely touches muscle at one
            # point won't cause an entire blob to be included.
            grown_mask = nama_candidates.copy()
            selem = morphology.disk(1)
            max_iterations = 50  # safety limit
            for _ in range(max_iterations):
                dilated = morphology.dilation(grown_mask, selem)
                new_lama = dilated & lama_candidates & ~grown_mask
                if not np.any(new_lama):
                    break
                grown_mask = grown_mask | new_lama

            # SMA = NAMA + grown LAMA
            self._sma_mask = grown_mask.astype(bool)

        return self._sma_mask
    
    def segment_nama(self) -> np.ndarray:
        """Segment Normal Attenuation Muscle Area (+30 to +150 HU).

        NAMA represents healthy muscle with minimal fat infiltration.
        Computed as a subset of SMA to ensure NAMA + LAMA = SMA.

        Returns:
            Binary mask of NAMA pixels
        """
        if self._nama_mask is None:
            sma_mask = self.segment_sma()
            self._nama_mask = (
                (self.hu_image >= self.thresholds.nama_min) &
                (self.hu_image <= self.thresholds.nama_max) &
                sma_mask
            )
        return self._nama_mask

    def segment_lama(self) -> np.ndarray:
        """Segment Low Attenuation Muscle Area (-29 to +29 HU).

        LAMA represents fat-infiltrated muscle (myosteatosis marker).
        Only includes LAMA pixels that are spatially connected to NAMA
        regions, ensuring they represent actual muscle tissue rather than
        non-muscle soft tissue with similar HU values.

        Returns:
            Binary mask of LAMA pixels
        """
        if self._lama_mask is None:
            sma_mask = self.segment_sma()
            self._lama_mask = (
                (self.hu_image >= self.thresholds.lama_min) &
                (self.hu_image <= self.thresholds.lama_max) &
                sma_mask
            )
        return self._lama_mask
    
    def segment_psoas(self) -> np.ndarray:
        """Segment psoas muscles (SMA tissue within the psoas zone).

        Psoas = muscle-density tissue (-29 to +150 HU) located within
        the bilateral psoas zones adjacent to the vertebral body.
        Psoas pixels are a SUBSET of SMA (they are included in SMA total).

        Returns:
            Binary mask of psoas muscle pixels
        """
        if self._psoas_mask is None:
            sma_mask = self.segment_sma()
            psoas_zone = self.muscle_compartment_generator.psoas_zone_mask
            if psoas_zone is not None:
                self._psoas_mask = sma_mask & psoas_zone
            else:
                self._psoas_mask = np.zeros_like(sma_mask)
        return self._psoas_mask

    def segment_imat(self) -> np.ndarray:
        """Segment Intermuscular Adipose Tissue (-190 to -30 HU).

        CRITICAL: Uses connectivity-based separation from MuscleCompartmentGenerator.
        Only fat that is NOT connected to the body outline is IMAT.

        Returns:
            Binary mask of IMAT pixels
        """
        if self._imat_mask is None:
            # Get IMAT directly from the connectivity-based analysis
            self._imat_mask = self.muscle_compartment_generator.get_imat_mask()
        return self._imat_mask
    
    def segment_sat(self) -> np.ndarray:
        """Segment Subcutaneous Adipose Tissue (-190 to -30 HU).
        
        SAT = fat connected to the body outline (outer fat layer).
        Uses connectivity-based separation from MuscleCompartmentGenerator.
        
        Returns:
            Binary mask of SAT pixels
        """
        if self._sat_mask is None:
            # Get SAT directly from the connectivity-based analysis
            self._sat_mask = self.muscle_compartment_generator.get_subcutaneous_fat_mask()
        return self._sat_mask
    
    def get_all_masks(self) -> Dict[str, np.ndarray]:
        """Get all tissue masks as a dictionary.

        Returns:
            Dictionary with keys 'nama', 'lama', 'imat', 'sma', 'psoas', 'sat'
        """
        return {
            'nama': self.segment_nama(),
            'lama': self.segment_lama(),
            'imat': self.segment_imat(),
            'sma': self.segment_sma(),
            'psoas': self.segment_psoas(),
            'sat': self.segment_sat()
        }


class AreaCalculator:
    """Calculates tissue areas from segmentation masks.
    
    This class converts pixel counts to physical area measurements (cm²)
    and computes clinical ratios for sarcopenia assessment.
    
    The area calculation uses:
        Area (cm²) = pixel_count * pixel_area_cm2
        
    Where pixel_area_cm2 is derived from DICOM PixelSpacing metadata.
    
    Attributes:
        masks: Dictionary of tissue masks
        pixel_area_cm2: Area of a single pixel in cm²
    """
    
    def __init__(self, 
                 masks: Dict[str, np.ndarray], 
                 pixel_area_cm2: float):
        """Initialize the area calculator.
        
        Args:
            masks: Dictionary of tissue masks (nama, lama, imat, sma, sat)
            pixel_area_cm2: Area of a single pixel in cm²
        """
        self.masks = masks
        self.pixel_area_cm2 = pixel_area_cm2
    
    def calculate_areas(self) -> TissueAreas:
        """Calculate tissue areas in cm².
        
        Returns:
            TissueAreas object with all calculated areas
        """
        nama_area = np.sum(self.masks['nama']) * self.pixel_area_cm2
        lama_area = np.sum(self.masks['lama']) * self.pixel_area_cm2
        imat_area = np.sum(self.masks['imat']) * self.pixel_area_cm2
        sma_area = np.sum(self.masks['sma']) * self.pixel_area_cm2
        psoas_area = np.sum(self.masks.get('psoas', np.zeros_like(self.masks['sma']))) * self.pixel_area_cm2
        sat_area = np.sum(self.masks.get('sat', np.zeros_like(self.masks['sma']))) * self.pixel_area_cm2

        # TAMA is typically synonymous with SMA at L3 level
        tama_area = sma_area

        return TissueAreas(
            nama=nama_area,
            lama=lama_area,
            imat=imat_area,
            sma=sma_area,
            tama=tama_area,
            psoas=psoas_area,
            sat=sat_area
        )
    
    def calculate_ratios(self, 
                         areas: TissueAreas,
                         height_m: float,
                         weight_kg: float) -> ClinicalRatios:
        """Calculate clinical ratios for sarcopenia assessment.
        
        Args:
            areas: TissueAreas object with calculated areas
            height_m: Patient height in meters
            weight_kg: Patient weight in kilograms
            
        Returns:
            ClinicalRatios object with all calculated ratios
        """
        # Calculate BMI
        bmi = weight_kg / (height_m ** 2)
        
        # Calculate ratios
        sma_bmi = areas.sma / bmi if bmi > 0 else 0
        nama_bmi = areas.nama / bmi if bmi > 0 else 0
        lama_bmi = areas.lama / bmi if bmi > 0 else 0
        
        # NAMA/TAMA ratio (indicator of muscle quality)
        # Higher ratio = healthier muscle (less fat infiltration)
        nama_tama = areas.nama / areas.tama if areas.tama > 0 else 0
        
        return ClinicalRatios(
            sma_bmi=sma_bmi,
            nama_bmi=nama_bmi,
            lama_bmi=lama_bmi,
            nama_tama=nama_tama
        )


class Visualizer:
    """Generates visualization overlays for body composition analysis.
    
    This class creates color-coded RGB overlays on the original CT image
    to visualize the tissue segmentation results.
    
    Color scheme:
    - NAMA (Red): Normal attenuation muscle
    - LAMA (Cyan): Low attenuation muscle
    - IMAT (Yellow): Intermuscular fat
    - SAT (Dark Green): Subcutaneous fat (optional display)
    
    Attributes:
        hu_image: Original image in Hounsfield Units
        masks: Dictionary of tissue masks
        colors: VisualizationColors object
    """
    
    def __init__(self,
                 hu_image: np.ndarray,
                 masks: Dict[str, np.ndarray],
                 colors: Optional[VisualizationColors] = None,
                 body_mask: Optional[np.ndarray] = None):
        """Initialize the visualizer.

        Args:
            hu_image: 2D numpy array of HU values
            masks: Dictionary of tissue masks
            colors: Custom colors (uses defaults if None)
            body_mask: Binary mask of the body region. Pixels outside
                       this mask are forced to black in the overlay.
        """
        self.hu_image = hu_image
        self.masks = masks
        self.colors = colors or VisualizationColors()
        self.body_mask = body_mask
    
    def generate_overlay(self,
                         window_center: int = 40,
                         window_width: int = 400,
                         show_sat: bool = False,
                         alpha: float = 1.0) -> np.ndarray:
        """Generate RGB overlay image with tissue colors.
        
        Args:
            window_center: HU value for window center (brightness)
            window_width: HU range for window (contrast)
            show_sat: Whether to show subcutaneous fat
            alpha: Opacity of tissue overlay (0-1)
            
        Returns:
            3-channel RGB numpy array
        """
        # Apply windowing to HU image
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        windowed = np.clip((self.hu_image - lower) / (upper - lower), 0, 1)
        
        # Create 3-channel RGB from grayscale
        rgb = np.stack([windowed] * 3, axis=-1)
        
        # Apply tissue colors with alpha blending.
        # Psoas is rendered LAST so it appears on top of NAMA/LAMA
        # (psoas pixels are a subset of SMA, so they overlap).
        tissue_layers = [('nama', self.colors.nama, True),
                         ('lama', self.colors.lama, True),
                         ('imat', self.colors.imat, True),
                         ('sat', self.colors.sat, show_sat),
                         ('psoas', self.colors.psoas, True)]

        for key, color, show in tissue_layers:
            if not show or key not in self.masks:
                continue
            mask = self.masks[key]
            if not np.any(mask):
                continue
            if alpha < 1.0:
                rgb[mask] = alpha * np.array(color) + (1 - alpha) * rgb[mask]
            else:
                rgb[mask] = color

        # Enforce body mask: force all non-body pixels to black
        if self.body_mask is not None:
            rgb[~self.body_mask] = 0.0

        return rgb
    
    def save_overlay(self, 
                     output_path: str,
                     dpi: int = 150,
                     show_sat: bool = False,
                     **kwargs) -> None:
        """Save the overlay visualization to a file.
        
        Args:
            output_path: Path to save the image
            dpi: Resolution in dots per inch
            show_sat: Whether to show subcutaneous fat
            **kwargs: Arguments passed to generate_overlay
        """
        import matplotlib.pyplot as plt
        
        overlay = self.generate_overlay(show_sat=show_sat, **kwargs)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title('L3 Body Composition Analysis')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='NAMA (Normal Muscle)'),
            Patch(facecolor='cyan', label='LAMA (Low Attenuation Muscle)'),
            Patch(facecolor='yellow', label='IMAT (Intermuscular Fat)')
        ]
        if show_sat:
            legend_elements.append(
                Patch(facecolor='green', label='SAT (Subcutaneous Fat)')
            )
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()


class L3Analyzer:
    """Main orchestrator class for L3 body composition analysis.
    
    VERSION 2.0: Now includes muscle compartment generation for accurate
    separation of subcutaneous fat from IMAT.
    
    This class provides a high-level interface that coordinates all
    analysis steps:
    1. DICOM loading and HU conversion
    2. Body mask generation (CT bed removal)
    3. Muscle compartment generation (NEW - separates SAT from IMAT)
    4. Tissue segmentation (NAMA, LAMA, IMAT, SMA)
    5. Area calculation
    6. Clinical ratio computation
    7. Visualization generation
    
    Example usage:
        analyzer = L3Analyzer("ct_slice.dcm")
        results = analyzer.analyze(height_m=1.75, weight_kg=70)
        analyzer.save_visualization("output.png")
    
    Attributes:
        dicom_loader: DICOMLoader instance
        body_mask_generator: BodyMaskGenerator instance
        muscle_compartment_generator: MuscleCompartmentGenerator instance
        segmenter: TissueSegmenter instance
        thresholds: Current HU thresholds
    """
    
    def __init__(self,
                 dicom_path: str,
                 thresholds: Optional[HUThresholds] = None,
                 file_type: str = 'dicom',
                 pixel_spacing: Tuple[float, float] = (1.0, 1.0),
                 hu_min: float = -160.0,
                 hu_max: float = 240.0):
        """Initialize the L3 analyzer.

        Args:
            dicom_path: Path to the image file (DICOM or JPEG)
            thresholds: Custom HU thresholds (uses validated defaults if None)
            file_type: 'dicom' or 'jpeg' to select the loader
            pixel_spacing: Pixel spacing in mm (only used for JPEG)
            hu_min: HU value for pixel 0 (only used for JPEG)
            hu_max: HU value for pixel 255 (only used for JPEG)
        """
        self.thresholds = thresholds or HUThresholds()

        # Load image based on file type
        if file_type == 'jpeg':
            self.dicom_loader = JPEGLoader(
                dicom_path,
                pixel_spacing=pixel_spacing,
                hu_min=hu_min,
                hu_max=hu_max
            )
            # For JPEG: use the pre-computed body mask from pixel-level detection
            # This bypasses BodyMaskGenerator which relies on HU thresholding
            # that doesn't work well for photos of monitors
            self._body_mask = self.dicom_loader.precomputed_body_mask
            self.body_mask_generator = None
        else:
            self.dicom_loader = DICOMLoader(dicom_path)
            # For DICOM: use standard HU-based body mask generation
            self.body_mask_generator = BodyMaskGenerator(
                self.dicom_loader.hu_image,
                self.thresholds.body_threshold
            )
            self._body_mask = self.body_mask_generator.generate_mask()
        
        # NEW: Generate muscle compartment mask using connectivity-based method
        self.muscle_compartment_generator = MuscleCompartmentGenerator(
            self.dicom_loader.hu_image,
            self._body_mask,
            self.thresholds
        )
        self._muscle_compartment_mask = self.muscle_compartment_generator.generate_muscle_compartment(
            method='connectivity'  # Use the improved connectivity-based method
        )
        
        # Initialize segmenter with muscle compartment generator
        self.segmenter = TissueSegmenter(
            self.dicom_loader.hu_image,
            self._body_mask,
            self.muscle_compartment_generator,  # Pass generator, not just mask
            self.thresholds
        )
        
        # Results storage
        self._masks: Optional[Dict[str, np.ndarray]] = None
        self._areas: Optional[TissueAreas] = None
        self._ratios: Optional[ClinicalRatios] = None
        self._visualizer: Optional[Visualizer] = None
    
    def analyze(self, 
                height_m: float, 
                weight_kg: float) -> Dict[str, Any]:
        """Perform complete L3 body composition analysis.
        
        Args:
            height_m: Patient height in meters
            weight_kg: Patient weight in kilograms
            
        Returns:
            Dictionary containing:
            - 'areas': TissueAreas object
            - 'ratios': ClinicalRatios object
            - 'metadata': DICOM metadata dictionary
            - 'bmi': Calculated BMI
        """
        # Segment all tissues
        self._masks = self.segmenter.get_all_masks()
        
        # Calculate areas
        calculator = AreaCalculator(
            self._masks,
            self.dicom_loader.pixel_area_cm2
        )
        self._areas = calculator.calculate_areas()
        
        # Calculate ratios
        self._ratios = calculator.calculate_ratios(
            self._areas,
            height_m,
            weight_kg
        )
        
        # Calculate BMI
        bmi = weight_kg / (height_m ** 2)
        
        # Initialize visualizer with body mask to enforce background = black
        self._visualizer = Visualizer(
            self.dicom_loader.hu_image,
            self._masks,
            body_mask=self._body_mask
        )
        
        return {
            'areas': self._areas,
            'ratios': self._ratios,
            'metadata': self.dicom_loader.get_metadata(),
            'bmi': bmi
        }
    
    def get_overlay_image(self, **kwargs) -> np.ndarray:
        """Get the RGB overlay image array.
        
        Args:
            **kwargs: Arguments passed to Visualizer.generate_overlay
            
        Returns:
            3-channel RGB numpy array
            
        Raises:
            RuntimeError: If analyze() hasn't been called
        """
        if self._visualizer is None:
            raise RuntimeError("Call analyze() before generating visualization")
        return self._visualizer.generate_overlay(**kwargs)
    
    def save_visualization(self, output_path: str, **kwargs) -> None:
        """Save the overlay visualization to a file.
        
        Args:
            output_path: Path to save the image
            **kwargs: Arguments passed to Visualizer.save_overlay
            
        Raises:
            RuntimeError: If analyze() hasn't been called
        """
        if self._visualizer is None:
            raise RuntimeError("Call analyze() before saving visualization")
        self._visualizer.save_overlay(output_path, **kwargs)
    
    def get_results_dict(self) -> Dict[str, float]:
        """Get analysis results as a flat dictionary.
        
        Useful for exporting to CSV or DataFrame.
        
        Returns:
            Dictionary with all numerical results
            
        Raises:
            RuntimeError: If analyze() hasn't been called
        """
        if self._areas is None or self._ratios is None:
            raise RuntimeError("Call analyze() before getting results")
        
        return {
            'NAMA_cm2': self._areas.nama,
            'LAMA_cm2': self._areas.lama,
            'IMAT_cm2': self._areas.imat,
            'SMA_cm2': self._areas.sma,
            'TAMA_cm2': self._areas.tama,
            'Psoas_cm2': self._areas.psoas,
            'SAT_cm2': self._areas.sat,
            'SMA_BMI': self._ratios.sma_bmi,
            'NAMA_BMI': self._ratios.nama_bmi,
            'LAMA_BMI': self._ratios.lama_bmi,
            'NAMA_TAMA': self._ratios.nama_tama
        }
    
    def update_thresholds(self, **kwargs) -> None:
        """Update HU thresholds and re-initialize segmenter.
        
        Args:
            **kwargs: Threshold values to update (e.g., nama_min=35)
        """
        # Update threshold values
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
        
        # Re-generate muscle compartment with new thresholds
        self.muscle_compartment_generator = MuscleCompartmentGenerator(
            self.dicom_loader.hu_image,
            self._body_mask,
            self.thresholds
        )
        self._muscle_compartment_mask = self.muscle_compartment_generator.generate_muscle_compartment(
            method='connectivity'
        )
        
        # Re-initialize segmenter with new thresholds
        self.segmenter = TissueSegmenter(
            self.dicom_loader.hu_image,
            self._body_mask,
            self.muscle_compartment_generator,  # Pass generator
            self.thresholds
        )
        
        # Clear cached results
        self._masks = None
        self._areas = None
        self._ratios = None
        self._visualizer = None


# Convenience function for quick analysis
def analyze_l3_slice(dicom_path: str,
                     height_m: float,
                     weight_kg: float,
                     output_image_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for single-slice analysis.
    
    Args:
        dicom_path: Path to DICOM file
        height_m: Patient height in meters
        weight_kg: Patient weight in kilograms
        output_image_path: Optional path to save visualization
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = L3Analyzer(dicom_path)
    results = analyzer.analyze(height_m, weight_kg)
    
    if output_image_path:
        analyzer.save_visualization(output_image_path)
    
    return results
