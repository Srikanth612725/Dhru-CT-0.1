# Auto-Sarcopenia L3 Analyst (Dhru-CT-1.1)

A research-grade Python library and Streamlit dashboard for automated body composition analysis of CT images at the L3 vertebra level. This tool replaces manual software like OsiriX by providing automated tissue partitioning and clinical ratio calculations.

## Overview

This project was developed to democratize medical imaging analysis by providing an open-source, Python-based alternative to proprietary CT analysis software. It automates the segmentation of muscle and adipose tissue at the third lumbar vertebra (L3) level, which is the standard anatomical landmark for assessing sarcopenia and myosteatosis.

## Features

**DICOM Processing**: Automated extraction of pixel data with proper Hounsfield Unit (HU) conversion using RescaleSlope and RescaleIntercept from DICOM metadata.

**Engineering-Led Cleaning**: Automatic removal of CT bed/table and external artifacts using morphological operations and Connected Component Analysis (CCA).

**Tissue Partitioning**: High-precision segmentation based on validated HU ranges:

| Tissue | HU Range | Description |
|--------|----------|-------------|
| NAMA | +30 to +150 | Normal Attenuation Muscle Area |
| LAMA | -29 to +29 | Low Attenuation Muscle Area |
| IMAT | -190 to -30 | Intermuscular Adipose Tissue |
| SMA | -29 to +150 | Total Skeletal Muscle Area |

**Clinical Metrics**: Calculation of BMI-normalized ratios including SMA/BMI, NAMA/BMI, LAMA/BMI, and NAMA/TAMA.

**Visualization**: Generation of color-coded RGB overlays with NAMA (Red), LAMA (Cyan), and IMAT (Yellow) highlighting.

**Interactive Dashboard**: Streamlit-based web interface with real-time threshold adjustment and visualization.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Dhru-CT-1.1.git
cd Dhru-CT-1.1

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line / Script Usage

```python
from l3_analyzer import L3Analyzer, analyze_l3_slice

# Quick analysis with convenience function
results = analyze_l3_slice(
    dicom_path="path/to/ct_slice.dcm",
    height_m=1.75,
    weight_kg=70,
    output_image_path="output_visualization.png"
)

# Access results
print(f"SMA: {results['areas'].sma:.2f} cm²")
print(f"NAMA/TAMA: {results['ratios'].nama_tama:.3f}")
```

### Full Analysis with Custom Thresholds

```python
from l3_analyzer import L3Analyzer, HUThresholds

# Create custom thresholds
thresholds = HUThresholds(
    nama_min=35,   # Adjust NAMA lower bound
    nama_max=150,
    lama_min=-29,
    lama_max=29,
    imat_min=-190,
    imat_max=-30
)

# Initialize analyzer
analyzer = L3Analyzer("ct_slice.dcm", thresholds)

# Run analysis
results = analyzer.analyze(height_m=1.75, weight_kg=70)

# Get numerical results
print(results['areas'])   # TissueAreas dataclass
print(results['ratios'])  # ClinicalRatios dataclass

# Generate and save visualization
analyzer.save_visualization("output.png")

# Export to dictionary (for CSV/DataFrame)
flat_results = analyzer.get_results_dict()
```

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

This launches an interactive web interface where you can upload DICOM files, adjust HU thresholds with sliders, and see real-time visualization updates.

## Architecture

The codebase follows a modular design with clear separation of concerns:

```
l3_analyzer.py
├── DICOMLoader          # DICOM parsing and HU conversion
├── BodyMaskGenerator    # CT bed removal and body isolation
├── TissueSegmenter      # HU-based tissue classification
├── AreaCalculator       # Physical area computation
├── Visualizer           # RGB overlay generation
└── L3Analyzer           # Main orchestrator class
```

## Clinical References

The HU thresholds used in this tool are based on validated research:

1. Aubrey J et al., "Measurement of skeletal muscle radiation attenuation and basis of its biological variation," Current Oncology, 2014
2. Prado CM et al., "Prevalence and clinical implications of sarcopenic obesity," Lancet Oncology, 2008
3. Martin L et al., "Cancer cachexia in the age of obesity," Journal of Clinical Oncology, 2013

## Clinical Significance

**SMA/BMI**: The primary metric for diagnosing sarcopenia. Lower values indicate reduced muscle mass relative to body size.

**NAMA/TAMA**: A key indicator of muscle quality. A low ratio suggests myosteatosis (fat infiltration in muscle tissue), which is associated with poor clinical outcomes even when total muscle area appears normal.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

BSD 3-Clause License - see LICENSE file for details.

## Acknowledgments

This project was developed as part of an effort to democratize medical imaging analysis tools for research purposes. Special thanks to the medical imaging research community for establishing and validating the HU threshold standards used in this tool.
