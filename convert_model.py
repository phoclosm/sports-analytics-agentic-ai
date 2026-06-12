"""
Model Converter for Windows Compatibility
Loads a fastai model and re-exports it for Windows systems
"""

import pathlib
import platform
import sys
from fastai.vision.all import *

def convert_model(input_path: str, output_path: str):
    """
    Load and re-save a fastai model for cross-platform compatibility
    """
    print(f"Converting model: {input_path}")
    print(f"Platform: {platform.system()}")
    
    # Apply Windows patch if needed
    temp = None
    if platform.system() == "Windows":
        print("Applying Windows compatibility patch...")
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        # Load the model
        print("Loading model...")
        learner = load_learner(input_path)
        
        # Restore path class
        if temp is not None:
            pathlib.PosixPath = temp
        
        # Re-export the model
        print(f"Exporting model to: {output_path}")
        learner.export(output_path)
        
        print("✅ Model converted successfully!")
        print(f"New model saved at: {output_path}")
        
        # Test loading the new model
        print("\nTesting new model...")
        test_learner = load_learner(output_path)
        print("✅ New model loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        if temp is not None:
            pathlib.PosixPath = temp
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_model.py <input_model.pkl> [output_model.pkl]")
        print("\nExample: python convert_model.py sports_classifier.pkl sports_classifier_win.pkl")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else "sports_classifier_win.pkl"
    
    success = convert_model(input_model, output_model)
    sys.exit(0 if success else 1)