"""
Enhanced Training script for ML-based symptoms diagnosis models
Supports multiple datasets and improved accuracy
"""

import sys
import json
import glob
from pathlib import Path
from backend.ml_diagnosis_enhanced import EnhancedMLDiagnosisModel

def main():
    """Main training function"""
    print("=" * 70)
    print("Enhanced ML Symptoms Diagnosis Model Training")
    print("=" * 70)
    
    # Initialize enhanced model
    model = EnhancedMLDiagnosisModel()
    
    # Collect dataset paths
    dataset_paths = []
    
    # Check command line arguments for dataset files
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.endswith('.json'):
                dataset_paths.append(arg)
            elif '*' in arg or '?' in arg:
                # Support glob patterns
                dataset_paths.extend(glob.glob(arg))
    
    # Look for dataset files in common locations
    if not dataset_paths:
        # Check for datasets directory
        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            json_files = list(datasets_dir.glob("*.json"))
            dataset_paths.extend([str(f) for f in json_files])
        
        # Check for example_dataset.json in root
        example_dataset = Path("example_dataset.json")
        if example_dataset.exists():
            dataset_paths.append(str(example_dataset))
    
    if dataset_paths:
        print(f"\nFound {len(dataset_paths)} dataset file(s):")
        for path in dataset_paths:
            print(f"  - {path}")
    else:
        print("\nNo dataset files found. Using expanded default training data...")
        print("To use custom datasets:")
        print("  1. Create a 'datasets' folder and add JSON files")
        print("  2. Or specify files: python train_models.py dataset1.json dataset2.json")
    
    # Train models
    print("\nStarting enhanced model training...")
    print("-" * 70)
    
    try:
        results = model.train(
            dataset_paths=dataset_paths if dataset_paths else None,
            use_cross_validation=True
        )
        
        print("\n" + "=" * 70)
        print("Training Results:")
        print("=" * 70)
        print(f"Urgency Model Accuracy: {results['urgency_accuracy']:.3f}")
        print(f"Urgency Model F1-Score: {results['urgency_f1']:.3f}")
        print(f"Cause Model Accuracy: {results['cause_accuracy']:.3f}")
        print(f"Cause Model F1-Score: {results['cause_f1']:.3f}")
        print(f"Training Samples: {results['training_samples']}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"Total Features: {results['total_features']}")
        print("\n[SUCCESS] Enhanced models trained and saved successfully!")
        print("You can now use the improved ML-based diagnosis in your application.")
        
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

