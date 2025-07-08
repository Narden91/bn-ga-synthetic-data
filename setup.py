"""
Setup script for the Bayesian Network GA-based Synthetic Data Generator.
Run this script to verify the installation and setup.
"""

import sys
import subprocess
import importlib


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.7+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
    return True


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def check_imports():
    """Check if all required modules can be imported."""
    required_modules = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('pgmpy', None),
        ('deap', None),
        ('sklearn', None)
    ]
    
    all_good = True
    print("\nChecking module imports...")
    
    for module, alias in required_modules:
        try:
            if alias:
                importlib.import_module(module)
                print(f"✅ {module} imported successfully.")
            else:
                importlib.import_module(module)
                print(f"✅ {module} imported successfully.")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            all_good = False
    
    return all_good


def check_data_file():
    """Check if data file exists."""
    import os
    data_file = "data/Dati_wallbox_aggregati.csv"
    
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        return True
    else:
        print(f"⚠️  Data file not found: {data_file}")
        print("   Please ensure your CSV file is in the data/ directory.")
        return False


def run_quick_test():
    """Run a quick functionality test."""
    print("\nRunning quick functionality test...")
    try:
        from main import run_quick_test
        run_quick_test()
        print("✅ Quick test completed successfully.")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    """Main setup and verification script."""
    print("="*60)
    print("BAYESIAN NETWORK GA SYNTHETIC DATA GENERATOR - SETUP")
    print("="*60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Check imports
    if not check_imports():
        success = False
    
    # Check data file
    data_exists = check_data_file()
    
    # Run quick test if everything is working
    if success:
        if not run_quick_test():
            success = False
    
    print("\n" + "="*60)
    if success:
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("\nYou can now run the main pipeline:")
        print("  python main.py")
        if not data_exists:
            print("\n⚠️  Remember to add your data file to the data/ directory.")
    else:
        print("❌ SETUP INCOMPLETE - Please fix the above issues.")
    
    print("="*60)


if __name__ == "__main__":
    main()
