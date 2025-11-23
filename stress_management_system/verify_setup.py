"""
Setup Verification Script
Checks if all components are properly installed and configured
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(name, status, message=""):
    """Print check result"""
    symbol = "✓" if status else "✗"
    status_text = "OK" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{symbol}{reset} {name}: {status_text}", end="")
    if message:
        print(f" - {message}")
    else:
        print()

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 8)
    status = version >= required
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    return status, message

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'tensorflow',
        'matplotlib', 'seaborn', 'plotly', 'shap', 'lime',
        'joblib', 'imblearn', 'chromadb', 'sentence_transformers'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'imblearn':
                __import__('imblearn')
            else:
                __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    return missing, installed

def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'data',
        'models',
        'results',
        'knowledge_base',
        'src'
    ]
    
    existing = []
    missing = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            existing.append(dir_name)
        else:
            missing.append(dir_name)
    
    return existing, missing

def check_source_files():
    """Check if source files exist"""
    required_files = [
        'src/data_preprocessing.py',
        'src/stress_detection_models.py',
        'src/explainable_ai.py',
        'src/generative_ai.py',
        'src/visualization.py',
        'src/pipeline.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    existing = []
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    return existing, missing

def check_wesad_dataset():
    """Check if WESAD dataset is available"""
    wesad_path = "data/WESAD"
    
    if not os.path.exists(wesad_path):
        return False, "WESAD directory not found"
    
    # Check for subject files
    subjects = [f'S{i}' for i in list(range(2, 12)) + list(range(13, 18))]
    found_subjects = []
    
    for subject in subjects:
        subject_file = os.path.join(wesad_path, subject, f"{subject}.pkl")
        if os.path.exists(subject_file):
            found_subjects.append(subject)
    
    if len(found_subjects) == 0:
        return False, "No subject files found"
    elif len(found_subjects) < len(subjects):
        return True, f"{len(found_subjects)}/{len(subjects)} subjects found"
    else:
        return True, f"All {len(subjects)} subjects found"

def check_processed_data():
    """Check if processed data exists"""
    processed_file = "data/processed_wesad.csv"
    
    if os.path.exists(processed_file):
        size_mb = os.path.getsize(processed_file) / 1024 / 1024
        return True, f"Found ({size_mb:.1f} MB)"
    else:
        return False, "Not found - needs processing"

def check_models():
    """Check if trained models exist"""
    model_files = [
        'models/random_forest_model.pkl',
        'models/svm_model.pkl',
        'models/scaler.pkl'
    ]
    
    found = []
    for model_file in model_files:
        if os.path.exists(model_file):
            found.append(os.path.basename(model_file))
    
    if len(found) == 0:
        return False, "No models found - needs training"
    elif len(found) < len(model_files):
        return True, f"{len(found)}/{len(model_files)} models found"
    else:
        return True, f"All models found"

def main():
    """Run all checks"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      AI-Driven Stress Management System - Setup Check       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    all_passed = True
    
    # Check Python version
    print_header("1. Python Environment")
    status, message = check_python_version()
    print_check("Python version (≥3.8)", status, message)
    if not status:
        all_passed = False
    
    # Check dependencies
    print_header("2. Python Dependencies")
    missing, installed = check_dependencies()
    
    if len(missing) == 0:
        print_check(f"All packages installed ({len(installed)} packages)", True)
    else:
        print_check("Package installation", False, f"{len(missing)} missing")
        print("\n  Missing packages:")
        for pkg in missing:
            print(f"    - {pkg}")
        print("\n  Install with: pip install -r requirements.txt")
        all_passed = False
    
    # Check directory structure
    print_header("3. Directory Structure")
    existing, missing = check_directory_structure()
    
    if len(missing) == 0:
        print_check("All directories present", True, f"{len(existing)} directories")
    else:
        print_check("Directory structure", False, f"{len(missing)} missing")
        print("\n  Missing directories:")
        for dir_name in missing:
            print(f"    - {dir_name}")
        all_passed = False
    
    # Check source files
    print_header("4. Source Files")
    existing, missing = check_source_files()
    
    if len(missing) == 0:
        print_check("All source files present", True, f"{len(existing)} files")
    else:
        print_check("Source files", False, f"{len(missing)} missing")
        print("\n  Missing files:")
        for file_path in missing:
            print(f"    - {file_path}")
        all_passed = False
    
    # Check WESAD dataset
    print_header("5. WESAD Dataset")
    status, message = check_wesad_dataset()
    print_check("WESAD dataset", status, message)
    
    if not status:
        print("\n  Download from: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset")
        print("  See DATASET_SETUP.md for instructions")
    
    # Check processed data
    print_header("6. Processed Data")
    status, message = check_processed_data()
    print_check("Processed data", status, message)
    
    if not status:
        print("\n  Process data with: python src/data_preprocessing.py")
        print("  Or run full pipeline: python src/pipeline.py")
    
    # Check models
    print_header("7. Trained Models")
    status, message = check_models()
    print_check("Trained models", status, message)
    
    if not status:
        print("\n  Train models with: python src/pipeline.py")
    
    # Final summary
    print_header("Summary")
    
    if all_passed:
        print("\n  ✅ All critical components are installed and configured!")
        print("\n  🚀 You're ready to run the system!")
        print("\n  Next steps:")
        print("    - Run demo: python demo.py")
        print("    - Run full pipeline: python src/pipeline.py")
        print("    - Read documentation: README.md")
    else:
        print("\n  ⚠️ Some components need attention (see details above)")
        print("\n  📋 Setup checklist:")
        print("    1. Install Python 3.8+")
        print("    2. Install dependencies: pip install -r requirements.txt")
        print("    3. Download WESAD dataset (see DATASET_SETUP.md)")
        print("    4. Run: python src/pipeline.py")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
