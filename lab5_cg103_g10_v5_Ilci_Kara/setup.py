import subprocess
import sys
import os


def setup_environment():
    print("Setting up virtual environment for the EARIN Lab 5 project...")

    # Create virtual environment
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path])
    else:
        print("Virtual environment already exists.")

    # Determine the pip path based on the operating system
    if os.name == "nt":  # Windows
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:  # macOS/Linux
        pip_path = os.path.join(venv_path, "bin", "pip")

    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])

    # Install required packages
    print("Installing required packages...")
    subprocess.run(
        [
            pip_path,
            "install",
            "torch",
            "torchvision",
            "matplotlib",
            "numpy",
            "pandas",
            "seaborn",
        ]
    )

    print("\nSetup complete! You can activate the virtual environment with:")
    if os.name == "nt":  # Windows
        print("venv\\Scripts\\activate")
    else:  # macOS/Linux
        print("source venv/bin/activate")

    print("\nAfter activation, run the scripts with:")
    print("python mlp_kmnist.py")
    print("python analyze_results.py")


if __name__ == "__main__":
    setup_environment()
