"""
Auto-Sarcopenia L3 Analyst - Streamlit Dashboard
=================================================
An interactive web interface for automated body composition analysis
of CT images at the L3 vertebra level.

Features:
- DICOM file upload
- Real-time tissue segmentation visualization
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
from pathlib import Path
from typing import Optional
import io

# Import our analysis module
from l3_analyzer import (
    L3Analyzer, 
    HUThresholds, 
    TissueAreas, 
    ClinicalRatios,
    DICOMLoader
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
            'overlay_alpha': overlay_alpha
        }


def render_file_upload():
    """Render the file upload section."""
    st.header("📁 Upload DICOM File")
    
    uploaded_file = st.file_uploader(
        "Choose a DICOM file (.dcm)",
        type=['dcm', 'dicom'],
        help="Upload a single CT slice at the L3 vertebra level"
    )
    
    return uploaded_file


def process_dicom(uploaded_file, params: dict) -> Optional[L3Analyzer]:
    """Process the uploaded DICOM file."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Create custom thresholds
        thresholds = HUThresholds(
            nama_min=params['nama_min'],
            nama_max=params['nama_max'],
            lama_min=params['lama_min'],
            lama_max=params['lama_max'],
            imat_min=params['imat_min'],
            imat_max=params['imat_max']
        )
        
        # Initialize analyzer
        analyzer = L3Analyzer(tmp_path, thresholds)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return analyzer
        
    except Exception as e:
        st.error(f"Error loading DICOM file: {str(e)}")
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


def render_info():
    """Render information about the analysis."""
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### Auto-Sarcopenia L3 Analyst
        
        This tool automates body composition analysis from CT images at the 
        third lumbar vertebra (L3) level, replacing manual software like OsiriX.
        
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


def main():
    """Main application entry point."""
    # Initialize
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar and get parameters
    params = render_sidebar()
    
    # Main content area
    uploaded_file = render_file_upload()
    
    if uploaded_file is not None:
        with st.spinner("Processing DICOM file..."):
            analyzer = process_dicom(uploaded_file, params)
        
        if analyzer is not None:
            # Run analysis
            with st.spinner("Analyzing tissue composition..."):
                results = analyzer.analyze(
                    height_m=params['height_m'],
                    weight_kg=params['weight_kg']
                )
            
            # Store in session state
            st.session_state.analyzer = analyzer
            st.session_state.results = results
            
            # Render all sections
            render_visualization(analyzer, params)
            st.divider()
            render_results(results, params)
            st.divider()
            render_export(analyzer, results)
    else:
        # Show placeholder
        st.info("👆 Upload a DICOM file to begin analysis")
        
        # Show demo info
        render_info()


if __name__ == "__main__":
    main()
