# run_full_pipeline.py

import os
import sys
import subprocess

# Setup Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

def run_script(script_path):
    """Helper function to run a script and print output."""
    print(f"\nRunning {script_path} ...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"{script_path} completed successfully.")
    else:
        print(f"{script_path} failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Script {script_path} failed.")

def main():
    # List of scripts to run in order
    scripts = [
        "scripts/train_model.py",
        "scripts/score_and_rank_customers.py",
        "scripts/cluster_customers.py",
        "scripts/cluster_rank_customers.py",
        "scripts/monitoring_check.py",
    ]

    for script in scripts:
        run_script(script)

    print("\nFull CLV Engine pipeline completed successfully!")

if __name__ == "__main__":
    main()
