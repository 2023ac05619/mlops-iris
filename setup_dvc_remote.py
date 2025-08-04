import subprocess
import sys
import os
from pathlib import Path

def setup_dvc_remote():
    """Setup DVC remote storage configuration."""
    
    print("Setting up DVC remote storage...")
    
    # Check if DVC is initialized
    if not Path(".dvc").exists():
        print("Initializing DVC...")
        subprocess.run(["dvc", "init"], check=True)
    
    # Configure remote storage
    repo_url = input("Enter your GitHub repository URL (e.g., https://github.com/<user name>/repo.git): ")
    
    try:
        # Add remote
        subprocess.run([
            "dvc", "remote", "add", "-d", "origin", repo_url
        ], check=True)
        
        # Configure authentication
        subprocess.run([
            "dvc", "remote", "modify", "origin", "auth", "github"
        ], check=True)
        
        print("DVC remote configured successfully!")
        print("Make sure to set GITHUB_TOKEN environment variable for authentication.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error configuring DVC remote: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_dvc_remote()