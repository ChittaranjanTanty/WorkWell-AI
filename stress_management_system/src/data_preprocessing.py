"""
Data Preprocessing Module for WESAD Dataset
Handles loading, cleaning, normalization, and feature extraction
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class WESADPreprocessor:
    """Preprocessor for WESAD dataset"""
    
    def __init__(self, data_path, window_size=60, overlap=30):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to WESAD dataset directory
            window_size: Window size in seconds for segmentation
            overlap: Overlap in seconds between windows
        """
        self.data_path = data_path
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def load_subject_data(self, subject_id):
        """
        Load data for a specific subject
        
        Args:
            subject_id: Subject ID (e.g., 'S2', 'S3', etc.)
            
        Returns:
            Dictionary containing subject data
        """
        subject_file = os.path.join(self.data_path, subject_id, f"{subject_id}.pkl")
        
        if not os.path.exists(subject_file):
            raise FileNotFoundError(f"Subject file not found: {subject_file}")
        
        with open(subject_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        return data
    
    def extract_chest_features(self, signal_data, sampling_rate=700):
        """
        Extract features from chest sensor signals
        
        Args:
            signal_data: Raw signal array
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['median'] = np.median(signal_data)
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Frequency domain features
        freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=min(256, len(signal_data)))
        features['psd_mean'] = np.mean(psd)
        features['psd_std'] = np.std(psd)
        features['dominant_freq'] = freqs[np.argmax(psd)]
        
        return features
    
    def extract_hrv_features(self, ecg_signal, sampling_rate=700):
        """
        Extract Heart Rate Variability (HRV) features from ECG
        
        Args:
            ecg_signal: ECG signal array
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of HRV features
        """
        features = {}
        
        try:
            # Simple peak detection (R-peaks in ECG)
            # Using a simple threshold-based method
            threshold = np.mean(ecg_signal) + 0.5 * np.std(ecg_signal)
            peaks, _ = signal.find_peaks(ecg_signal, height=threshold, distance=sampling_rate//3)
            
            if len(peaks) > 1:
                # RR intervals (time between peaks)
                rr_intervals = np.diff(peaks) / sampling_rate * 1000  # in ms
                
                # Time domain HRV features
                features['hr_mean'] = 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
                features['hr_std'] = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
                features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0
                features['sdnn'] = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
                features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100 if len(rr_intervals) > 1 else 0
            else:
                features['hr_mean'] = 0
                features['hr_std'] = 0
                features['rmssd'] = 0
                features['sdnn'] = 0
                features['pnn50'] = 0
        except Exception as e:
            print(f"HRV extraction error: {e}")
            features = {'hr_mean': 0, 'hr_std': 0, 'rmssd': 0, 'sdnn': 0, 'pnn50': 0}
        
        return features
    
    def segment_signal(self, signal_data, labels, sampling_rate=700):
        """
        Segment signal into windows
        
        Args:
            signal_data: Multi-channel signal array
            labels: Label array
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of segmented windows and corresponding labels
        """
        window_samples = int(self.window_size * sampling_rate)
        overlap_samples = int(self.overlap * sampling_rate)
        step_size = window_samples - overlap_samples
        
        segments = []
        segment_labels = []
        
        for start in range(0, len(signal_data) - window_samples + 1, step_size):
            end = start + window_samples
            segment = signal_data[start:end]
            
            # Get most common label in window
            window_labels = labels[start:end]
            label = np.bincount(window_labels).argmax()
            
            # Only keep baseline (1), stress (2), and amusement (3)
            # Filter out meditation (4) and transient states (0, 5, 6, 7)
            if label in [1, 2, 3]:
                segments.append(segment)
                # Remap labels: 1->0 (Baseline), 2->1 (Stress), 3->2 (Amusement)
                segment_labels.append(label - 1)
        
        return segments, segment_labels
    
    def process_subject(self, subject_id):
        """
        Process all data for a single subject
        
        Args:
            subject_id: Subject ID
            
        Returns:
            DataFrame with features and labels
        """
        print(f"Processing subject {subject_id}...")
        
        try:
            data = self.load_subject_data(subject_id)
        except FileNotFoundError:
            print(f"Skipping {subject_id} - file not found")
            return None
        
        # Extract chest sensor data
        chest_data = data['signal']['chest']
        labels = data['label']
        
        # Get individual signals
        ecg = chest_data['ECG'].flatten()
        eda = chest_data['EDA'].flatten()
        emg = chest_data['EMG'].flatten()
        temp = chest_data['Temp'].flatten()
        resp = chest_data['Resp'].flatten()
        
        # Ensure all signals have same length
        min_len = min(len(ecg), len(eda), len(emg), len(temp), len(resp), len(labels))
        ecg = ecg[:min_len]
        eda = eda[:min_len]
        emg = emg[:min_len]
        temp = temp[:min_len]
        resp = resp[:min_len]
        labels = labels[:min_len]
        
        # Segment signals
        ecg_segments, seg_labels = self.segment_signal(ecg.reshape(-1, 1), labels)
        eda_segments, _ = self.segment_signal(eda.reshape(-1, 1), labels)
        emg_segments, _ = self.segment_signal(emg.reshape(-1, 1), labels)
        temp_segments, _ = self.segment_signal(temp.reshape(-1, 1), labels)
        resp_segments, _ = self.segment_signal(resp.reshape(-1, 1), labels)
        
        # Extract features for each segment
        feature_list = []
        
        for i in range(len(ecg_segments)):
            segment_features = {}
            
            # ECG features + HRV
            ecg_feats = self.extract_chest_features(ecg_segments[i].flatten())
            hrv_feats = self.extract_hrv_features(ecg_segments[i].flatten())
            for k, v in ecg_feats.items():
                segment_features[f'ecg_{k}'] = v
            for k, v in hrv_feats.items():
                segment_features[f'hrv_{k}'] = v
            
            # EDA features
            eda_feats = self.extract_chest_features(eda_segments[i].flatten())
            for k, v in eda_feats.items():
                segment_features[f'eda_{k}'] = v
            
            # EMG features
            emg_feats = self.extract_chest_features(emg_segments[i].flatten())
            for k, v in emg_feats.items():
                segment_features[f'emg_{k}'] = v
            
            # Temperature features
            temp_feats = self.extract_chest_features(temp_segments[i].flatten())
            for k, v in temp_feats.items():
                segment_features[f'temp_{k}'] = v
            
            # Respiration features
            resp_feats = self.extract_chest_features(resp_segments[i].flatten())
            for k, v in resp_feats.items():
                segment_features[f'resp_{k}'] = v
            
            # Add label
            segment_features['label'] = seg_labels[i]
            segment_features['subject'] = subject_id
            
            feature_list.append(segment_features)
        
        df = pd.DataFrame(feature_list)
        print(f"Extracted {len(df)} segments from {subject_id}")
        
        return df
    
    def process_all_subjects(self, subject_ids=None):
        """
        Process all subjects in dataset
        
        Args:
            subject_ids: List of subject IDs to process (default: S2-S17)
            
        Returns:
            Combined DataFrame with all features
        """
        if subject_ids is None:
            # WESAD dataset has subjects S2 to S17
            subject_ids = [f'S{i}' for i in range(2, 18)]
        
        all_data = []
        
        for subject_id in subject_ids:
            subject_df = self.process_subject(subject_id)
            if subject_df is not None:
                all_data.append(subject_df)
        
        if len(all_data) == 0:
            raise ValueError("No data was processed. Check if WESAD dataset is properly extracted.")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nTotal segments: {len(combined_df)}")
        print(f"Label distribution:\n{combined_df['label'].value_counts()}")
        
        return combined_df
    
    def normalize_features(self, df, fit=True):
        """
        Normalize features using StandardScaler
        
        Args:
            df: DataFrame with features
            fit: Whether to fit scaler (True for training, False for test)
            
        Returns:
            DataFrame with normalized features
        """
        feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    def save_processed_data(self, df, output_path):
        """Save processed data to CSV"""
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    
    def save_scaler(self, scaler_path):
        """Save scaler for later use"""
        import joblib
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "../data/WESAD"
    
    preprocessor = WESADPreprocessor(data_path, window_size=60, overlap=30)
    
    # Process all subjects
    df = preprocessor.process_all_subjects()
    
    # Normalize features
    df = preprocessor.normalize_features(df)
    
    # Save processed data
    preprocessor.save_processed_data(df, "../data/processed_wesad.csv")
    preprocessor.save_scaler("../models/scaler.pkl")
