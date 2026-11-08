from pathlib import Path
import importlib
import platform
import sys


ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "data/raw",
    "data/processed",
    "data/synthetic",
    "notebooks",
    "models",
    "artifacts",
    "reports/figures",
    "synthetic_data",
    "fine_tuning",
    "genai",
    "agents",
    "business",
    "dashboard",
    "tests",
    "docs",
]

PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "xgboost",
    "shap",
    "matplotlib",
    "seaborn",
    "plotly",
    "streamlit",
    "requests",
    "joblib",
    "dotenv",
]

STARTER_FILES = {
    "README.md": """# DefenseAI

AI-Powered Equipment Intelligence, Predictive Maintenance
and Readiness Platform.

## Initial scope

- Aircraft-engine degradation analysis
- Remaining useful life prediction
- Explainable machine learning
- Maintenance documentation retrieval
- Human-reviewed maintenance recommendations
- Readiness and financial scenario simulation

## Data transparency

This portfolio project will use public research datasets
and explicitly labeled simulated operational scenarios.

It does not use classified military data.
It is not certified for operational maintenance decisions.

## Current stage

Project environment and folder setup.
""",
    ".gitignore": """# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/

# Secrets
.env
.env.*
!.env.example

# Environments
.venv/
venv/

# Local data and generated artifacts
data/raw/*
data/processed/*
data/synthetic/*
artifacts/*
reports/figures/*

# Preserve directory placeholders
!.gitkeep

# Local editor and OS files
.vscode/
.idea/
.DS_Store
Thumbs.db
""",
    ".env.example": """# Add configuration when the relevant feature is implemented.
# Never commit real credentials.
""",
    "docs/project_scope.md": """# Initial Project Scope

## First milestone
Load and validate the NASA C-MAPSS FD001 dataset.

## Second milestone
Create engine-level train/validation splits and RUL labels.

## Third milestone
Train and evaluate baseline and XGBoost models.

## Later milestones
Explainability, dashboard, RAG, fine-tuning,
agent workflows, and a readiness simulator.

## Evaluation boundaries
Separate observed data, model estimates, and simulation assumptions.
Do not describe simulated savings as measured operational savings.
""",
}


def create_structure():
    for folder in FOLDERS:
        directory = ROOT / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)

    for relative_path, content in STARTER_FILES.items():
        destination = ROOT / relative_path
        if not destination.exists():
            destination.write_text(content, encoding="utf-8")
            print(f"Created: {relative_path}")
        else:
            print(f"Kept existing file: {relative_path}")


def verify_environment():
    print("\nEnvironment")
    print("-" * 55)
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Project directory: {ROOT}")

    failures = []

    print("\nPackage checks")
    print("-" * 55)

    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "import successful")
            print(f"[OK] {package}: {version}")
        except Exception as exc:
            failures.append(package)
            print(f"[FAILED] {package}: {type(exc).__name__}: {exc}")

    print("\n" + "-" * 55)

    if failures:
        print("Setup requires attention.")
        print("Failed package checks:", ", ".join(failures))
        raise SystemExit(1)

    print("SUCCESS: DefenseAI setup completed.")
    print("All initial packages imported successfully.")
    print("Next milestone: download and validate NASA C-MAPSS FD001.")


if __name__ == "__main__":
    create_structure()
    verify_environment()