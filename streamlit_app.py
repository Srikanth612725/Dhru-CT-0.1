"""
Auto-Sarcopenia L3 Analyst - Streamlit Dashboard
=================================================
An interactive web interface for automated body composition analysis
of CT images at the L3 vertebra level.

Features:
- Single-slice analysis (DICOM or JPEG)
- Full 3D volumetric analysis (DICOM series)
- Real-time tissue segmentation visualization
- Slice browser for volumetric data
- Adjustable HU threshold sliders
- Clinical ratio calculations
- Export results to CSV

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import zipfile
from pathlib import Path
from typing import Optional, List
import io

# Import our analysis modules
from l3_analyzer import (
    L3Analyzer,
    HUThresholds,
    TissueAreas,
    ClinicalRatios,
    DICOMLoader,
    JPEGLoader
)
from volumetric_analyzer import (
    DICOMSeriesLoader,
    VolumetricAnalyzer,
    TissueVolumes,
    VolumetricRatios,
)

# Page configuration
st.set_page_config(
    page_title="L3 Analyst-Dhru_CT-1.1",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'dicom_loaded' not in st.session_state:
        st.session_state.dicom_loaded = False
    if 'thresholds' not in st.session_state:
        st.session_state.thresholds = HUThresholds()
    if 'vol_analyzer' not in st.session_state:
        st.session_state.vol_analyzer = None
    if 'vol_results' not in st.session_state:
        st.session_state.vol_results = None


def render_header():
    """Render the application header."""
    st.markdown('<p class="main-header">🏥 Auto-Sarcopenia L3 Analyst</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Automated body composition analysis of CT images '
        'at the L3 vertebra level</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.header("📋 Patient Information")

        height_cm = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=170.0,
            step=1.0,
            help="Patient height in centimeters"
        )

        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0,
            step=0.5,
            help="Patient weight in kilograms"
        )

        st.divider()

        st.header("🎚️ HU Thresholds")
        st.caption("Adjust thresholds for tissue segmentation")

        # NAMA thresholds
        st.subheader("NAMA (Normal Muscle)")
        nama_min = st.slider(
            "NAMA Min (HU)",
            min_value=0,
            max_value=100,
            value=30,
            help="Lower bound for Normal Attenuation Muscle Area"
        )
        nama_max = st.slider(
            "NAMA Max (HU)",
            min_value=100,
            max_value=200,
            value=150,
            help="Upper bound for Normal Attenuation Muscle Area"
        )

        # LAMA thresholds
        st.subheader("LAMA (Low Attenuation Muscle)")
        lama_min = st.slider(
            "LAMA Min (HU)",
            min_value=-100,
            max_value=0,
            value=-29,
            help="Lower bound for Low Attenuation Muscle Area"
        )
        lama_max = st.slider(
            "LAMA Max (HU)",
            min_value=0,
            max_value=50,
            value=29,
            help="Upper bound for Low Attenuation Muscle Area"
        )

        # IMAT thresholds
        st.subheader("IMAT (Intermuscular Fat)")
        imat_min = st.slider(
            "IMAT Min (HU)",
            min_value=-250,
            max_value=-100,
            value=-190,
            help="Lower bound for Intermuscular Adipose Tissue"
        )
        imat_max = st.slider(
            "IMAT Max (HU)",
            min_value=-100,
            max_value=0,
            value=-30,
            help="Upper bound for Intermuscular Adipose Tissue"
        )

        st.divider()

        # Visualization settings
        st.header("🎨 Visualization")
        window_center = st.slider(
            "Window Center (HU)",
            min_value=-200,
            max_value=200,
            value=40,
            help="Center of the HU display window"
        )
        window_width = st.slider(
            "Window Width (HU)",
            min_value=100,
            max_value=800,
            value=400,
            help="Width of the HU display window"
        )
        overlay_alpha = st.slider(
            "Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Opacity of tissue color overlay"
        )

        st.divider()

        # JPEG-specific settings
        st.header("🖼️ JPEG Settings")
        st.caption("Only used when uploading JPEG images")

        pixel_spacing_row = st.number_input(
            "Pixel Spacing - Row (mm)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help="Physical pixel spacing in mm (row direction). Check your CT scanner or DICOM viewer for the correct value."
        )
        pixel_spacing_col = st.number_input(
            "Pixel Spacing - Column (mm)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help="Physical pixel spacing in mm (column direction)."
        )
        hu_min = st.number_input(
            "HU Min (pixel 0)",
            min_value=-2000.0,
            max_value=0.0,
            value=-160.0,
            step=10.0,
            help="Hounsfield Unit value corresponding to the darkest pixel (0). Default assumes abdominal CT window (center=40, width=400)."
        )
        hu_max = st.number_input(
            "HU Max (pixel 255)",
            min_value=0.0,
            max_value=3000.0,
            value=240.0,
            step=10.0,
            help="Hounsfield Unit value corresponding to the brightest pixel (255)."
        )

        return {
            'height_m': height_cm / 100.0,
            'weight_kg': weight_kg,
            'nama_min': nama_min,
            'nama_max': nama_max,
            'lama_min': lama_min,
            'lama_max': lama_max,
            'imat_min': imat_min,
            'imat_max': imat_max,
            'window_center': window_center,
            'window_width': window_width,
            'overlay_alpha': overlay_alpha,
            'pixel_spacing_row': pixel_spacing_row,
            'pixel_spacing_col': pixel_spacing_col,
            'hu_min': hu_min,
            'hu_max': hu_max
        }


# ---------------------------------------------------------------------------
# Single-slice helpers (unchanged)
# ---------------------------------------------------------------------------

def render_file_upload():
    """Render the file upload section."""
    st.header("📁 Upload CT Image")

    uploaded_file = st.file_uploader(
        "Choose a DICOM (.dcm) or JPEG (.jpg/.jpeg) file",
        type=['dcm', 'dicom', 'jpg', 'jpeg'],
        help="Upload a single CT slice at the L3 vertebra level"
    )

    return uploaded_file


def _detect_file_type(uploaded_file) -> str:
    """Detect whether the uploaded file is DICOM or JPEG."""
    name = uploaded_file.name.lower()
    if name.endswith(('.jpg', '.jpeg')):
        return 'jpeg'
    return 'dicom'


def process_uploaded_file(uploaded_file, params: dict) -> Optional[L3Analyzer]:
    """Process the uploaded DICOM or JPEG file."""
    file_type = _detect_file_type(uploaded_file)
    suffix = '.jpg' if file_type == 'jpeg' else '.dcm'

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        thresholds = HUThresholds(
            nama_min=params['nama_min'],
            nama_max=params['nama_max'],
            lama_min=params['lama_min'],
            lama_max=params['lama_max'],
            imat_min=params['imat_min'],
            imat_max=params['imat_max']
        )

        if file_type == 'jpeg':
            analyzer = L3Analyzer(
                tmp_path,
                thresholds,
                file_type='jpeg',
                pixel_spacing=(
                    params.get('pixel_spacing_row', 1.0),
                    params.get('pixel_spacing_col', 1.0)
                ),
                hu_min=params.get('hu_min', -160.0),
                hu_max=params.get('hu_max', 240.0)
            )
        else:
            analyzer = L3Analyzer(tmp_path, thresholds)

        os.unlink(tmp_path)
        return analyzer

    except Exception as e:
        st.error(f"Error loading {file_type.upper()} file: {str(e)}")
        return None


def render_results(results: dict, params: dict):
    """Render the analysis results."""
    areas: TissueAreas = results['areas']
    ratios: ClinicalRatios = results['ratios']
    metadata = results['metadata']

    # Patient info section
    st.header("👤 Patient Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patient ID", metadata.get('patient_id', 'N/A'))
    with col2:
        st.metric("Study Date", metadata.get('study_date', 'N/A'))
    with col3:
        st.metric("BMI", f"{results['bmi']:.1f} kg/m²")
    with col4:
        st.metric("Pixel Spacing",
                  f"{metadata['pixel_spacing'][0]:.2f} mm")

    st.divider()

    # Tissue areas section
    st.header("📊 Tissue Areas (cm²)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🔴 NAMA",
            f"{areas.nama:.2f}",
            help="Normal Attenuation Muscle Area"
        )
    with col2:
        st.metric(
            "🔵 LAMA",
            f"{areas.lama:.2f}",
            help="Low Attenuation Muscle Area"
        )
    with col3:
        st.metric(
            "🟡 IMAT",
            f"{areas.imat:.2f}",
            help="Intermuscular Adipose Tissue"
        )
    with col4:
        st.metric(
            "📐 SMA (Total)",
            f"{areas.sma:.2f}",
            help="Total Skeletal Muscle Area"
        )

    st.divider()

    # Clinical ratios section
    st.header("📈 Clinical Ratios")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "SMA/BMI",
            f"{ratios.sma_bmi:.2f}",
            help="Skeletal Muscle Area normalized by BMI"
        )
    with col2:
        st.metric(
            "NAMA/BMI",
            f"{ratios.nama_bmi:.2f}",
            help="Normal Muscle Area normalized by BMI"
        )
    with col3:
        st.metric(
            "LAMA/BMI",
            f"{ratios.lama_bmi:.2f}",
            help="Low Attenuation Muscle normalized by BMI"
        )
    with col4:
        # NAMA/TAMA ratio with color indicator
        nama_tama = ratios.nama_tama
        status = "🟢" if nama_tama > 0.5 else "🟡" if nama_tama > 0.3 else "🔴"
        st.metric(
            f"NAMA/TAMA {status}",
            f"{nama_tama:.3f}",
            help="Muscle quality indicator (higher is better)"
        )


def render_visualization(analyzer: L3Analyzer, params: dict):
    """Render the visualization section."""
    st.header("🖼️ Tissue Segmentation Visualization")

    # Generate overlay image
    overlay = analyzer.get_overlay_image(
        window_center=params['window_center'],
        window_width=params['window_width'],
        alpha=params['overlay_alpha']
    )

    # Create two columns: image and legend
    col1, col2 = st.columns([3, 1])

    with col1:
        st.image(
            overlay,
            caption="L3 Body Composition Analysis",
            use_container_width=True
        )

    with col2:
        st.subheader("Legend")
        st.markdown("🔴 **NAMA** - Normal Attenuation Muscle")
        st.markdown("🔵 **LAMA** - Low Attenuation Muscle")
        st.markdown("🟡 **IMAT** - Intermuscular Fat")

        st.divider()

        st.subheader("HU Ranges")
        st.markdown(f"NAMA: {params['nama_min']} to {params['nama_max']} HU")
        st.markdown(f"LAMA: {params['lama_min']} to {params['lama_max']} HU")
        st.markdown(f"IMAT: {params['imat_min']} to {params['imat_max']} HU")


def render_export(analyzer: L3Analyzer, results: dict):
    """Render the export section."""
    st.header("💾 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        # Export as CSV
        results_dict = analyzer.get_results_dict()
        df = pd.DataFrame([results_dict])
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="l3_analysis_results.csv",
            mime="text/csv"
        )

    with col2:
        # Export visualization
        import matplotlib.pyplot as plt

        overlay = analyzer.get_overlay_image()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title('L3 Body Composition Analysis')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='NAMA (Normal Muscle)'),
            Patch(facecolor='cyan', label='LAMA (Low Attenuation)'),
            Patch(facecolor='yellow', label='IMAT (Fat)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        st.download_button(
            label="📥 Download Image (PNG)",
            data=buf,
            file_name="l3_visualization.png",
            mime="image/png"
        )


# ---------------------------------------------------------------------------
# Volume analysis helpers
# ---------------------------------------------------------------------------

def _extract_dicom_files_from_upload(uploaded_files) -> List[str]:
    """Save uploaded DICOM files to a temp directory and return paths.

    Handles both raw .dcm uploads and .zip archives.
    """
    tmp_dir = tempfile.mkdtemp(prefix="dicom_series_")
    paths: List[str] = []

    for uf in uploaded_files:
        name_lower = uf.name.lower()

        if name_lower.endswith('.zip'):
            # Extract ZIP contents
            zip_path = os.path.join(tmp_dir, uf.name)
            with open(zip_path, 'wb') as f:
                f.write(uf.getvalue())
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)
            os.unlink(zip_path)
            # Walk extracted tree for DICOM files
            for root, _dirs, files in os.walk(tmp_dir):
                for fname in files:
                    fl = fname.lower()
                    if fl.endswith(('.dcm', '.dicom')) or not os.path.splitext(fl)[1]:
                        paths.append(os.path.join(root, fname))
        else:
            dest = os.path.join(tmp_dir, uf.name)
            with open(dest, 'wb') as f:
                f.write(uf.getvalue())
            paths.append(dest)

    return paths


def render_volume_upload():
    """Render the volume upload section."""
    st.header("📁 Upload DICOM Series")
    st.caption(
        "Upload all DICOM files from a single CT scan. You can upload "
        "individual .dcm files or a .zip archive containing the series."
    )

    uploaded_files = st.file_uploader(
        "Choose DICOM files or a ZIP archive",
        type=['dcm', 'dicom', 'zip'],
        accept_multiple_files=True,
        help="Upload all slices from a CT series (any number of files)",
        key="volume_uploader",
    )

    return uploaded_files


def process_volume_upload(uploaded_files, params: dict):
    """Load a DICOM series and run volumetric analysis."""
    try:
        dicom_paths = _extract_dicom_files_from_upload(uploaded_files)
        if not dicom_paths:
            st.error("No DICOM files found in upload.")
            return None, None

        st.info(f"Found **{len(dicom_paths)}** DICOM files. Loading series...")

        series = DICOMSeriesLoader(dicom_paths)

        st.success(
            f"Series loaded: **{series.num_slices}** slices, "
            f"slice thickness = {series.slice_thickness:.2f} mm, "
            f"pixel spacing = {series.pixel_spacing[0]:.2f} x {series.pixel_spacing[1]:.2f} mm"
        )

        thresholds = HUThresholds(
            nama_min=params['nama_min'],
            nama_max=params['nama_max'],
            lama_min=params['lama_min'],
            lama_max=params['lama_max'],
            imat_min=params['imat_min'],
            imat_max=params['imat_max'],
        )

        vol_analyzer = VolumetricAnalyzer(series, thresholds)

        progress_bar = st.progress(0, text="Analyzing slices...")

        def _update(current, total):
            progress_bar.progress(
                current / total,
                text=f"Analyzing slice {current} / {total}..."
            )

        results = vol_analyzer.analyze(
            height_m=params['height_m'],
            weight_kg=params['weight_kg'],
            progress_callback=_update,
        )

        progress_bar.progress(1.0, text="Analysis complete!")
        return vol_analyzer, results

    except Exception as e:
        st.error(f"Error processing DICOM series: {e}")
        return None, None


def render_volume_results(results: dict):
    """Render volumetric analysis results."""
    volumes: TissueVolumes = results['volumes']
    ratios: VolumetricRatios = results['ratios']
    metadata = results['metadata']

    # Patient info
    st.header("👤 Patient / Series Information")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Patient ID", metadata.get('patient_id', 'N/A'))
    with c2:
        st.metric("Study Date", metadata.get('study_date', 'N/A'))
    with c3:
        st.metric("Slices", metadata['num_slices'])
    with c4:
        st.metric("Slice Thickness", f"{metadata['slice_thickness_mm']:.2f} mm")
    with c5:
        st.metric("BMI", f"{results['bmi']:.1f} kg/m²")

    st.divider()

    # Tissue volumes
    st.header("📊 Tissue Volumes (cm³)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔴 NAMA", f"{volumes.nama:.2f}", help="Normal Attenuation Muscle Volume")
    with c2:
        st.metric("🔵 LAMA", f"{volumes.lama:.2f}", help="Low Attenuation Muscle Volume")
    with c3:
        st.metric("🟡 IMAT", f"{volumes.imat:.2f}", help="Intermuscular Adipose Tissue Volume")
    with c4:
        st.metric("📐 SMV (Total)", f"{volumes.sma:.2f}", help="Total Skeletal Muscle Volume")

    st.divider()

    # Volumetric ratios
    st.header("📈 Volumetric Clinical Ratios")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("SMV/BMI", f"{ratios.smv_bmi:.2f}", help="Skeletal Muscle Volume / BMI")
    with c2:
        st.metric("NAMA_vol/BMI", f"{ratios.nama_vol_bmi:.2f}")
    with c3:
        st.metric("LAMA_vol/BMI", f"{ratios.lama_vol_bmi:.2f}")
    with c4:
        nv = ratios.nama_tama_vol
        status = "🟢" if nv > 0.5 else "🟡" if nv > 0.3 else "🔴"
        st.metric(f"NAMA/TAMA {status}", f"{nv:.3f}", help="Volumetric muscle quality")


def render_slice_browser(vol_analyzer: VolumetricAnalyzer, params: dict):
    """Render an interactive slice browser."""
    st.header("🔬 Slice Browser")

    num_slices = len(vol_analyzer.slice_results)
    slice_idx = st.slider(
        "Slice",
        min_value=0,
        max_value=num_slices - 1,
        value=num_slices // 2,
        help="Scroll through slices in the volume"
    )

    sr = vol_analyzer.slice_results[slice_idx]
    loc_str = f" (location: {sr.slice_location:.1f} mm)" if sr.slice_location is not None else ""

    overlay = vol_analyzer.get_slice_overlay(
        slice_idx,
        window_center=params['window_center'],
        window_width=params['window_width'],
        alpha=params['overlay_alpha'],
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(overlay, caption=f"Slice {slice_idx + 1} / {num_slices}{loc_str}", use_container_width=True)
    with col2:
        st.subheader("Slice Areas (cm²)")
        st.metric("NAMA", f"{sr.areas.nama:.2f}")
        st.metric("LAMA", f"{sr.areas.lama:.2f}")
        st.metric("IMAT", f"{sr.areas.imat:.2f}")
        st.metric("SMA", f"{sr.areas.sma:.2f}")

        st.divider()
        st.subheader("Legend")
        st.markdown("🔴 **NAMA** - Normal Muscle")
        st.markdown("🔵 **LAMA** - Low Attenuation Muscle")
        st.markdown("🟡 **IMAT** - Intermuscular Fat")


def render_per_slice_chart(vol_analyzer: VolumetricAnalyzer):
    """Render a chart showing tissue areas across all slices."""
    st.header("📉 Per-Slice Tissue Distribution")

    df = vol_analyzer.get_per_slice_dataframe()

    # Use slice_location for x-axis if available, else slice_index
    x_col = 'slice_index'
    x_label = 'Slice Index'
    if df['slice_location'].notna().all():
        x_col = 'slice_location'
        x_label = 'Slice Location (mm)'

    chart_df = df.set_index(x_col)[['NAMA_cm2', 'LAMA_cm2', 'IMAT_cm2', 'SMA_cm2']]
    st.line_chart(chart_df)
    st.caption(f"X-axis: {x_label}  |  Y-axis: Area (cm²)")


def render_volume_export(vol_analyzer: VolumetricAnalyzer, results: dict):
    """Render export buttons for volumetric results."""
    st.header("💾 Export Volumetric Results")

    col1, col2 = st.columns(2)

    with col1:
        # Summary CSV
        summary = vol_analyzer.get_results_dict()
        summary_df = pd.DataFrame([summary])
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Volume Summary (CSV)",
            data=csv_summary,
            file_name="volumetric_summary.csv",
            mime="text/csv",
            key="vol_summary_csv",
        )

    with col2:
        # Per-slice CSV
        per_slice_df = vol_analyzer.get_per_slice_dataframe()
        csv_slices = per_slice_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Per-Slice Data (CSV)",
            data=csv_slices,
            file_name="per_slice_areas.csv",
            mime="text/csv",
            key="vol_perslice_csv",
        )


# ---------------------------------------------------------------------------
# Info / About
# ---------------------------------------------------------------------------

def render_info():
    """Render information about the analysis."""
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### Auto-Sarcopenia L3 Analyst

        This tool automates body composition analysis from CT images at the
        third lumbar vertebra (L3) level, replacing manual software like OsiriX.

        #### Analysis Modes

        | Mode | Input | Output |
        |------|-------|--------|
        | **Single Slice** | One DICOM or JPEG file | Tissue areas (cm²) |
        | **3D Volume** | Full DICOM series (any number of slices) | Tissue volumes (cm³) + per-slice areas |

        #### Tissue Types

        | Tissue | HU Range | Description |
        |--------|----------|-------------|
        | **NAMA** | +30 to +150 | Normal Attenuation Muscle Area - healthy muscle |
        | **LAMA** | -29 to +29 | Low Attenuation Muscle Area - fat-infiltrated muscle |
        | **IMAT** | -190 to -30 | Intermuscular Adipose Tissue - fat between muscles |
        | **SMA** | -29 to +150 | Total Skeletal Muscle Area (NAMA + LAMA) |

        #### Clinical Significance

        - **SMA/BMI**: Primary metric for sarcopenia diagnosis
        - **NAMA/TAMA**: Key indicator of muscle quality (myosteatosis)
          - Higher values indicate healthier muscle composition
          - Lower values suggest fat infiltration

        #### References

        1. Aubrey J et al., Current Oncology, 2014
        2. Prado CM et al., Lancet Oncol, 2008
        3. Martin L et al., J Clin Oncol, 2013
        """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""
    initialize_session_state()
    render_header()
    params = render_sidebar()

    # Two tabs: single-slice and volumetric
    tab_single, tab_volume = st.tabs(["🔬 Single Slice Analysis", "🧊 3D Volume Analysis"])

    # ---- Single-slice tab ----
    with tab_single:
        uploaded_file = render_file_upload()

        if uploaded_file is not None:
            file_type = _detect_file_type(uploaded_file)

            if file_type == 'jpeg':
                st.warning(
                    "**JPEG Mode:** HU values are approximate (mapped from pixel "
                    "intensity). Area measurements require correct pixel spacing "
                    "in the sidebar. For clinical-grade results, use DICOM files."
                )

            spinner_msg = "Processing JPEG file..." if file_type == 'jpeg' else "Processing DICOM file..."
            with st.spinner(spinner_msg):
                analyzer = process_uploaded_file(uploaded_file, params)

            if analyzer is not None:
                with st.spinner("Analyzing tissue composition..."):
                    results = analyzer.analyze(
                        height_m=params['height_m'],
                        weight_kg=params['weight_kg']
                    )

                st.session_state.analyzer = analyzer
                st.session_state.results = results

                render_visualization(analyzer, params)
                st.divider()
                render_results(results, params)
                st.divider()
                render_export(analyzer, results)
        else:
            st.info("👆 Upload a DICOM or JPEG file to begin single-slice analysis")
            render_info()

    # ---- Volume tab ----
    with tab_volume:
        uploaded_files = render_volume_upload()

        if uploaded_files:
            vol_analyzer, vol_results = process_volume_upload(uploaded_files, params)

            if vol_analyzer is not None and vol_results is not None:
                st.session_state.vol_analyzer = vol_analyzer
                st.session_state.vol_results = vol_results

                render_volume_results(vol_results)
                st.divider()
                render_slice_browser(vol_analyzer, params)
                st.divider()
                render_per_slice_chart(vol_analyzer)
                st.divider()
                render_volume_export(vol_analyzer, vol_results)
        else:
            st.info(
                "👆 Upload a full DICOM series (all .dcm files or a .zip archive) "
                "to perform 3D volumetric analysis"
            )


if __name__ == "__main__":
    main()
