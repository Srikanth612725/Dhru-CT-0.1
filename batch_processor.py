"""
Batch Processing Utility for L3 Analysis
=========================================
Process multiple DICOM files and export results to CSV/Excel.

This module enables high-throughput analysis of patient cohorts,
which is a key advantage over manual software like OsiriX.

Author: Auto-Sarcopenia L3 Analyst Project
License: BSD 3-Clause
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from l3_analyzer import L3Analyzer, HUThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatientData:
    """Container for patient demographic data."""
    
    def __init__(self, 
                 patient_id: str,
                 dicom_path: str,
                 height_m: float,
                 weight_kg: float,
                 **additional_fields):
        """Initialize patient data.
        
        Args:
            patient_id: Unique patient identifier
            dicom_path: Path to the DICOM file
            height_m: Patient height in meters
            weight_kg: Patient weight in kilograms
            **additional_fields: Any additional metadata to include
        """
        self.patient_id = patient_id
        self.dicom_path = dicom_path
        self.height_m = height_m
        self.weight_kg = weight_kg
        self.additional_fields = additional_fields


class BatchProcessor:
    """Process multiple DICOM files for L3 body composition analysis.
    
    This class enables efficient batch processing of patient cohorts,
    with support for parallel processing, progress tracking, and
    comprehensive error handling.
    
    Example usage:
        processor = BatchProcessor()
        patients = processor.load_patient_list("patients.csv")
        results = processor.process_all(patients)
        processor.export_results(results, "output_results.csv")
    """
    
    def __init__(self, 
                 thresholds: Optional[HUThresholds] = None,
                 save_visualizations: bool = False,
                 output_dir: Optional[str] = None):
        """Initialize the batch processor.
        
        Args:
            thresholds: Custom HU thresholds (uses defaults if None)
            save_visualizations: Whether to save visualization images
            output_dir: Directory for saving visualizations
        """
        self.thresholds = thresholds or HUThresholds()
        self.save_visualizations = save_visualizations
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        
        if self.save_visualizations:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_patient_list(self, 
                          csv_path: str,
                          id_column: str = 'patient_id',
                          dicom_column: str = 'dicom_path',
                          height_column: str = 'height_m',
                          weight_column: str = 'weight_kg') -> List[PatientData]:
        """Load patient list from a CSV file.
        
        Args:
            csv_path: Path to CSV file with patient information
            id_column: Column name for patient ID
            dicom_column: Column name for DICOM file path
            height_column: Column name for height (in meters)
            weight_column: Column name for weight (in kg)
            
        Returns:
            List of PatientData objects
        """
        df = pd.read_csv(csv_path)
        
        required_columns = [id_column, dicom_column, height_column, weight_column]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        patients = []
        for _, row in df.iterrows():
            additional = {
                k: v for k, v in row.items() 
                if k not in required_columns
            }
            patients.append(PatientData(
                patient_id=str(row[id_column]),
                dicom_path=row[dicom_column],
                height_m=float(row[height_column]),
                weight_kg=float(row[weight_column]),
                **additional
            ))
        
        logger.info(f"Loaded {len(patients)} patients from {csv_path}")
        return patients
    
    def process_single(self, patient: PatientData) -> Dict:
        """Process a single patient's DICOM file.
        
        Args:
            patient: PatientData object with patient information
            
        Returns:
            Dictionary with patient ID and analysis results
        """
        result = {
            'patient_id': patient.patient_id,
            'status': 'success',
            'error': None
        }
        
        try:
            # Initialize analyzer
            analyzer = L3Analyzer(patient.dicom_path, self.thresholds)
            
            # Run analysis
            analysis_results = analyzer.analyze(
                height_m=patient.height_m,
                weight_kg=patient.weight_kg
            )
            
            # Get flat results dictionary
            metrics = analyzer.get_results_dict()
            result.update(metrics)
            
            # Add BMI
            result['BMI'] = analysis_results['bmi']
            
            # Add patient demographics
            result['height_m'] = patient.height_m
            result['weight_kg'] = patient.weight_kg
            result.update(patient.additional_fields)
            
            # Save visualization if requested
            if self.save_visualizations:
                viz_path = self.output_dir / f"{patient.patient_id}_l3_analysis.png"
                analyzer.save_visualization(str(viz_path))
                result['visualization_path'] = str(viz_path)
            
            logger.info(f"Successfully processed patient {patient.patient_id}")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Error processing patient {patient.patient_id}: {e}")
        
        return result
    
    def process_all(self, 
                    patients: List[PatientData],
                    parallel: bool = False,
                    max_workers: int = 4,
                    show_progress: bool = True) -> pd.DataFrame:
        """Process all patients in the list.
        
        Args:
            patients: List of PatientData objects
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with all analysis results
        """
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_single, p): p 
                    for p in patients
                }
                
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(patients), 
                                    desc="Processing patients")
                
                for future in iterator:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        patient = futures[future]
                        results.append({
                            'patient_id': patient.patient_id,
                            'status': 'error',
                            'error': str(e)
                        })
        else:
            iterator = patients
            if show_progress:
                iterator = tqdm(patients, desc="Processing patients")
            
            for patient in iterator:
                result = self.process_single(patient)
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Reorder columns
        priority_cols = ['patient_id', 'status', 'BMI', 
                         'NAMA_cm2', 'LAMA_cm2', 'IMAT_cm2', 'SMA_cm2',
                         'SMA_BMI', 'NAMA_BMI', 'LAMA_BMI', 'NAMA_TAMA']
        existing_priority = [c for c in priority_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in priority_cols]
        df = df[existing_priority + other_cols]
        
        # Summary statistics
        success_count = (df['status'] == 'success').sum()
        error_count = (df['status'] == 'error').sum()
        logger.info(f"Batch processing complete: {success_count} successful, "
                    f"{error_count} errors")
        
        return df
    
    def export_results(self, 
                       df: pd.DataFrame, 
                       output_path: str,
                       format: str = 'csv') -> None:
        """Export results to file.
        
        Args:
            df: DataFrame with analysis results
            output_path: Path to save the results
            format: Output format ('csv' or 'excel')
        """
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results exported to {output_path}")
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from batch results.
        
        Args:
            df: DataFrame with analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        successful = df[df['status'] == 'success']
        
        numeric_cols = ['NAMA_cm2', 'LAMA_cm2', 'IMAT_cm2', 'SMA_cm2',
                        'SMA_BMI', 'NAMA_BMI', 'LAMA_BMI', 'NAMA_TAMA', 'BMI']
        
        summary = {
            'total_patients': len(df),
            'successful': len(successful),
            'errors': len(df) - len(successful),
            'statistics': {}
        }
        
        for col in numeric_cols:
            if col in successful.columns:
                summary['statistics'][col] = {
                    'mean': successful[col].mean(),
                    'std': successful[col].std(),
                    'min': successful[col].min(),
                    'max': successful[col].max(),
                    'median': successful[col].median()
                }
        
        return summary


def main():
    """Example usage of batch processor."""
    # Example: Create a sample patient list
    sample_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'dicom_path': ['data/p001.dcm', 'data/p002.dcm', 'data/p003.dcm'],
        'height_m': [1.75, 1.68, 1.82],
        'weight_kg': [70, 65, 85],
        'age': [45, 52, 38],
        'gender': ['M', 'F', 'M']
    })
    
    # Save sample CSV
    sample_data.to_csv('sample_patient_list.csv', index=False)
    print("Sample patient list saved to 'sample_patient_list.csv'")
    print("\nTo use the batch processor:")
    print("""
    from batch_processor import BatchProcessor
    
    processor = BatchProcessor(save_visualizations=True, output_dir='./results')
    patients = processor.load_patient_list('patient_list.csv')
    results_df = processor.process_all(patients, parallel=True)
    processor.export_results(results_df, 'analysis_results.csv')
    """)


if __name__ == "__main__":
    main()
