"""
Quick Installation Test
Tests if all packages are properly installed
"""

import sys

print("=" * 70)
print("  AI-DRIVEN STRESS MANAGEMENT SYSTEM - Installation Test")
print("=" * 70)
print()

print("Testing package imports...\n")

packages_to_test = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn'),
    ('scipy', 'SciPy'),
    ('tensorflow', 'TensorFlow'),
    ('keras', 'Keras'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('shap', 'SHAP'),
    ('lime', 'LIME'),
    ('plotly', 'Plotly'),
    ('joblib', 'Joblib'),
    ('imblearn', 'Imbalanced-learn'),
    ('langchain', 'LangChain'),
    ('chromadb', 'ChromaDB'),
    ('sentence_transformers', 'Sentence Transformers'),
    ('transformers', 'Transformers'),
    ('torch', 'PyTorch'),
]

success_count = 0
failed_packages = []

for module_name, display_name in packages_to_test:
    try:
        __import__(module_name)
        print(f"✓ {display_name:<25} OK")
        success_count += 1
    except Exception as e:
        print(f"✗ {display_name:<25} FAILED: {str(e)[:50]}")
        failed_packages.append(display_name)

print()
print("=" * 70)
print(f"Results: {success_count}/{len(packages_to_test)} packages imported successfully")
print("=" * 70)

if failed_packages:
    print(f"\n⚠️ Failed packages: {', '.join(failed_packages)}")
    print("Try: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✅ All packages installed correctly!")
    print("\n📋 Next Steps:")
    print("   1. Download WESAD dataset from Kaggle")
    print("   2. Extract to: data/WESAD/")
    print("   3. Run: python src/pipeline.py")
    print("\nSee DATASET_SETUP.md for detailed instructions.")
    print()

# Test version information
print("\n" + "=" * 70)
print("  Package Versions")
print("=" * 70)

try:
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    
    print(f"Python:       {sys.version.split()[0]}")
    print(f"NumPy:        {np.__version__}")
    print(f"Pandas:       {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"TensorFlow:   {tf.__version__}")
    
except Exception as e:
    print(f"Error getting versions: {e}")

print("=" * 70)
print()
