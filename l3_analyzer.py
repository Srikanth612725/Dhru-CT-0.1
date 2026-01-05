"""
L3 Body Composition Analyzer
============================
A research-grade Python module for automated body composition analysis 
of CT images at the L3 vertebra level.

This module replaces manual software like OsiriX by providing automated 
tissue partitioning and clinical ratio calculations.

Clinical References for HU Ranges:
- Aubrey et al. (2014) Cancer Cachexia Study - NAMA/LAMA thresholds
- Prado et al. (2008) Sarcopenia Definition - SMA ranges
- Martin et al. (2013) Myosteatosis Assessment - IMAT definition

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
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
    """
    nama: float
    lama: float
    imat: float
    sma: float
    tama: float


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
        closed_mask = morphology.binary_closing(initial_mask, selem)
        
        # Step 3: Fill holes in the body mask
        # Important for including air-filled structures inside the body
        filled_mask = ndimage.binary_fill_holes(closed_mask)
        
        # Step 4: Remove small objects (noise, artifacts)
        cleaned_mask = morphology.remove_small_objects(
            filled_mask, 
            min_size=min_body_size
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
        body_mask = morphology.binary_opening(body_mask, morphology.disk(2))
        body_mask = morphology.binary_closing(body_mask, morphology.disk(2))
        
        self._body_mask = body_mask.astype(bool)
        return self._body_mask
    
    @property
    def body_mask(self) -> np.ndarray:
        """Get the generated body mask."""
        if self._body_mask is None:
            return self.generate_mask()
        return self._body_mask


class TissueSegmenter:
    """Segments tissues based on Hounsfield Unit thresholds.
    
    This class performs radiodensity-based muscle partitioning by
    binning pixels into clinically validated HU ranges:
    
    - NAMA (Normal Attenuation Muscle Area): +30 to +150 HU
      Represents healthy muscle with minimal fat infiltration
      
    - LAMA (Low Attenuation Muscle Area): -29 to +29 HU
      Represents muscle with significant lipid content (myosteatosis)
      
    - IMAT (Intermuscular Adipose Tissue): -190 to -30 HU
      Fat tissue located between muscle groups
      
    - SMA (Total Skeletal Muscle Area): -29 to +150 HU
      Sum of NAMA and LAMA
    
    Clinical Reference:
        Mourtzakis M et al., Appl Physiol Nutr Metab, 2008
        Prado CM et al., Lancet Oncol, 2008
    
    Attributes:
        hu_image: Input image in Hounsfield Units
        body_mask: Binary mask defining valid analysis region
        thresholds: HUThresholds object defining tissue boundaries
    """
    
    def __init__(self, 
                 hu_image: np.ndarray, 
                 body_mask: np.ndarray,
                 thresholds: Optional[HUThresholds] = None):
        """Initialize the tissue segmenter.
        
        Args:
            hu_image: 2D numpy array of HU values
            body_mask: Binary mask where True indicates valid pixels
            thresholds: Custom HU thresholds (uses defaults if None)
        """
        self.hu_image = hu_image
        self.body_mask = body_mask
        self.thresholds = thresholds or HUThresholds()
        
        # Tissue masks (computed on demand)
        self._nama_mask: Optional[np.ndarray] = None
        self._lama_mask: Optional[np.ndarray] = None
        self._imat_mask: Optional[np.ndarray] = None
        self._sma_mask: Optional[np.ndarray] = None
    
    def _create_tissue_mask(self, 
                            hu_min: int, 
                            hu_max: int,
                            apply_body_mask: bool = True,
                            min_region_size: int = 50) -> np.ndarray:
        """Create a binary mask for pixels within specified HU range.
        
        Args:
            hu_min: Minimum HU value (inclusive)
            hu_max: Maximum HU value (inclusive)
            apply_body_mask: Whether to restrict to body region
            min_region_size: Minimum pixel count for valid regions
            
        Returns:
            Binary numpy array of tissue mask
        """
        # Apply HU thresholding
        mask = (self.hu_image >= hu_min) & (self.hu_image <= hu_max)
        
        # Restrict to body region if requested
        if apply_body_mask:
            mask = mask & self.body_mask
        
        # Remove small isolated regions (noise)
        mask = morphology.remove_small_objects(mask, min_size=min_region_size)
        
        return mask.astype(bool)
    
    def segment_nama(self) -> np.ndarray:
        """Segment Normal Attenuation Muscle Area (+30 to +150 HU).
        
        Returns:
            Binary mask of NAMA pixels
        """
        if self._nama_mask is None:
            self._nama_mask = self._create_tissue_mask(
                self.thresholds.nama_min,
                self.thresholds.nama_max
            )
        return self._nama_mask
    
    def segment_lama(self) -> np.ndarray:
        """Segment Low Attenuation Muscle Area (-29 to +29 HU).
        
        Returns:
            Binary mask of LAMA pixels
        """
        if self._lama_mask is None:
            self._lama_mask = self._create_tissue_mask(
                self.thresholds.lama_min,
                self.thresholds.lama_max
            )
        return self._lama_mask
    
    def segment_imat(self) -> np.ndarray:
        """Segment Intermuscular Adipose Tissue (-190 to -30 HU).
        
        Returns:
            Binary mask of IMAT pixels
        """
        if self._imat_mask is None:
            self._imat_mask = self._create_tissue_mask(
                self.thresholds.imat_min,
                self.thresholds.imat_max
            )
        return self._imat_mask
    
    def segment_sma(self) -> np.ndarray:
        """Segment Total Skeletal Muscle Area (-29 to +150 HU).
        
        This is the union of NAMA and LAMA regions.
        
        Returns:
            Binary mask of SMA pixels
        """
        if self._sma_mask is None:
            self._sma_mask = self._create_tissue_mask(
                self.thresholds.sma_min,
                self.thresholds.sma_max
            )
        return self._sma_mask
    
    def get_all_masks(self) -> Dict[str, np.ndarray]:
        """Get all tissue masks as a dictionary.
        
        Returns:
            Dictionary with keys 'nama', 'lama', 'imat', 'sma'
        """
        return {
            'nama': self.segment_nama(),
            'lama': self.segment_lama(),
            'imat': self.segment_imat(),
            'sma': self.segment_sma()
        }


class AreaCalculator:
    """Calculates tissue areas and clinical ratios.
    
    Converts pixel counts to physical areas (cm²) using DICOM
    pixel spacing metadata, then computes BMI-normalized ratios
    for clinical assessment.
    
    Attributes:
        masks: Dictionary of tissue masks from TissueSegmenter
        pixel_area_cm2: Physical area of each pixel in cm²
    """
    
    def __init__(self, 
                 masks: Dict[str, np.ndarray], 
                 pixel_area_cm2: float):
        """Initialize the area calculator.
        
        Args:
            masks: Dictionary containing tissue masks
            pixel_area_cm2: Area of a single pixel in cm²
        """
        self.masks = masks
        self.pixel_area_cm2 = pixel_area_cm2
    
    def calculate_areas(self) -> TissueAreas:
        """Calculate absolute tissue areas in cm².
        
        Returns:
            TissueAreas dataclass with all area measurements
        """
        nama_area = np.sum(self.masks['nama']) * self.pixel_area_cm2
        lama_area = np.sum(self.masks['lama']) * self.pixel_area_cm2
        imat_area = np.sum(self.masks['imat']) * self.pixel_area_cm2
        sma_area = np.sum(self.masks['sma']) * self.pixel_area_cm2
        
        # TAMA is typically equivalent to SMA at L3 level
        tama_area = sma_area
        
        return TissueAreas(
            nama=nama_area,
            lama=lama_area,
            imat=imat_area,
            sma=sma_area,
            tama=tama_area
        )
    
    def calculate_ratios(self, 
                         areas: TissueAreas,
                         height_m: float,
                         weight_kg: float) -> ClinicalRatios:
        """Calculate BMI-normalized clinical ratios.
        
        Args:
            areas: TissueAreas object with absolute areas
            height_m: Patient height in meters
            weight_kg: Patient weight in kilograms
            
        Returns:
            ClinicalRatios dataclass with normalized ratios
        """
        # Calculate BMI
        bmi = weight_kg / (height_m ** 2)
        
        # Calculate normalized ratios
        sma_bmi = areas.sma / bmi if bmi > 0 else 0.0
        nama_bmi = areas.nama / bmi if bmi > 0 else 0.0
        lama_bmi = areas.lama / bmi if bmi > 0 else 0.0
        
        # NAMA/TAMA ratio (muscle quality indicator)
        nama_tama = areas.nama / areas.tama if areas.tama > 0 else 0.0
        
        return ClinicalRatios(
            sma_bmi=sma_bmi,
            nama_bmi=nama_bmi,
            lama_bmi=lama_bmi,
            nama_tama=nama_tama
        )


class Visualizer:
    """Generates color-coded overlay visualizations of tissue segmentation.
    
    Creates RGB images with tissue regions highlighted using clinically
    standard color coding:
    - NAMA: Red (healthy muscle)
    - LAMA: Cyan (fat-infiltrated muscle)
    - IMAT: Yellow (intermuscular fat)
    
    Attributes:
        hu_image: Original HU image for background
        masks: Dictionary of tissue masks
        colors: VisualizationColors object
    """
    
    def __init__(self, 
                 hu_image: np.ndarray,
                 masks: Dict[str, np.ndarray],
                 colors: Optional[VisualizationColors] = None):
        """Initialize the visualizer.
        
        Args:
            hu_image: 2D numpy array of HU values
            masks: Dictionary containing tissue masks
            colors: Custom colors (uses defaults if None)
        """
        self.hu_image = hu_image
        self.masks = masks
        self.colors = colors or VisualizationColors()
    
    def _normalize_image(self, 
                         window_center: int = 40,
                         window_width: int = 400) -> np.ndarray:
        """Apply windowing and normalize image to [0, 1] range.
        
        Args:
            window_center: Center of the HU window
            window_width: Width of the HU window
            
        Returns:
            Normalized 2D array in [0, 1] range
        """
        # Calculate window bounds
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2
        
        # Apply windowing and normalize
        normalized = np.clip(
            (self.hu_image - window_min) / (window_max - window_min),
            0, 1
        )
        return normalized
    
    def generate_overlay(self,
                         window_center: int = 40,
                         window_width: int = 400,
                         alpha: float = 0.7) -> np.ndarray:
        """Generate RGB overlay image with tissue segmentation.
        
        Args:
            window_center: HU window center for background
            window_width: HU window width for background
            alpha: Opacity of color overlay (0-1)
            
        Returns:
            3-channel RGB numpy array in [0, 1] range
        """
        # Create grayscale background
        gray = self._normalize_image(window_center, window_width)
        
        # Convert to 3-channel RGB
        rgb = np.stack([gray, gray, gray], axis=-1)
        
        # Apply tissue colors with blending
        # Order matters: IMAT first (yellow), then LAMA (cyan), then NAMA (red)
        # This ensures NAMA is most visible as it represents healthy muscle
        
        if 'imat' in self.masks:
            imat_mask = self.masks['imat']
            for i, c in enumerate(self.colors.imat):
                rgb[imat_mask, i] = alpha * c + (1 - alpha) * rgb[imat_mask, i]
        
        if 'lama' in self.masks:
            lama_mask = self.masks['lama']
            for i, c in enumerate(self.colors.lama):
                rgb[lama_mask, i] = alpha * c + (1 - alpha) * rgb[lama_mask, i]
        
        if 'nama' in self.masks:
            nama_mask = self.masks['nama']
            for i, c in enumerate(self.colors.nama):
                rgb[nama_mask, i] = alpha * c + (1 - alpha) * rgb[nama_mask, i]
        
        return rgb
    
    def save_overlay(self, 
                     output_path: str,
                     dpi: int = 150,
                     **kwargs) -> None:
        """Save the overlay visualization to a file.
        
        Args:
            output_path: Path to save the image
            dpi: Resolution in dots per inch
            **kwargs: Additional arguments for generate_overlay
        """
        import matplotlib.pyplot as plt
        
        overlay = self.generate_overlay(**kwargs)
        
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
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()


class L3Analyzer:
    """Main orchestrator class for L3 body composition analysis.
    
    This class provides a high-level interface that coordinates all
    analysis steps:
    1. DICOM loading and HU conversion
    2. Body mask generation (CT bed removal)
    3. Tissue segmentation (NAMA, LAMA, IMAT, SMA)
    4. Area calculation
    5. Clinical ratio computation
    6. Visualization generation
    
    Example usage:
        analyzer = L3Analyzer("ct_slice.dcm")
        results = analyzer.analyze(height_m=1.75, weight_kg=70)
        analyzer.save_visualization("output.png")
    
    Attributes:
        dicom_loader: DICOMLoader instance
        body_mask_generator: BodyMaskGenerator instance
        segmenter: TissueSegmenter instance
        thresholds: Current HU thresholds
    """
    
    def __init__(self, 
                 dicom_path: str,
                 thresholds: Optional[HUThresholds] = None):
        """Initialize the L3 analyzer.
        
        Args:
            dicom_path: Path to the DICOM file
            thresholds: Custom HU thresholds (uses validated defaults if None)
        """
        self.thresholds = thresholds or HUThresholds()
        
        # Load DICOM
        self.dicom_loader = DICOMLoader(dicom_path)
        
        # Generate body mask
        self.body_mask_generator = BodyMaskGenerator(
            self.dicom_loader.hu_image,
            self.thresholds.body_threshold
        )
        self._body_mask = self.body_mask_generator.generate_mask()
        
        # Initialize segmenter
        self.segmenter = TissueSegmenter(
            self.dicom_loader.hu_image,
            self._body_mask,
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
        
        # Initialize visualizer
        self._visualizer = Visualizer(
            self.dicom_loader.hu_image,
            self._masks
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
        
        # Re-initialize segmenter with new thresholds
        self.segmenter = TissueSegmenter(
            self.dicom_loader.hu_image,
            self._body_mask,
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
