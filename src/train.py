import subprocess
import sys

def run_local_pipeline():
    print("--- Running DVC pipeline locally ---")
    try:
        subprocess.run(["dvc", "repro"], check=True)
        print("--- DVC pipeline finished successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running DVC pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'dvc' command not found. Is DVC installed?", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_local_pipeline()
