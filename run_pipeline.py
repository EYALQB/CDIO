import subprocess
import sys
from pathlib import Path

# utilidats

def run_step(script_path, step_name):
    print(f"\n Executant {step_name}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"{step_name} completat correctament.\n")
    except subprocess.CalledProcessError:
        print(f"Error en {step_name}. Revisa la sortida anterior.\n")
        sys.exit(1)

# main

def main():
    base = Path(__file__).parent / "src"

    steps = [
        ("Pas 1 - Data Cleaning", base / "data_preparation.py"),
        ("Pas 2 - Exploratory Analysis", base / "exploratory_analysis.py"),
        ("Pas 3 - Model Fitting", base / "model_fitting.py"),
        ("Pas 4 - Model Evaluation", base / "model_evaluation.py"),
        ("Pas 5 - Forecasting", base / "forecasting.py"),
        ("Pas 6 - Validation & Discussion", base / "validation_discussion.py"),
    ]

    print("\nEXECUTEM EL PROJECTE")

    for name, script in steps:
        if not script.exists():
            print(f"No se encontró {script}, se omite.")
            continue
        run_step(str(script), name)

    print("\nPipeline acabat.")
    print("Resultats disponibles a la carpeta 'outputs/'.\n")

if __name__ == "__main__":
    main()
