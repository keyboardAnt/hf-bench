import subprocess
import os

import fire

def list_tracked_files(dirpath):
    # Run git ls-tree command and capture output
    cmd = ['git', 'ls-tree', '-r', 'origin/main', '--name-only', dirpath]
    result = subprocess.run(cmd, 
                            capture_output=True, 
                            text=True, 
                            check=True)
    # Split output into list of files
    files = result.stdout.strip().split('\n')
    # Filter out empty strings
    files = [f for f in files if f]
    return files


def main(dirpath: str):
    filepaths = list_tracked_files(dirpath)
    for f in filepaths:
        print(f)
    

if __name__ == "__main__":
    fire.Fire(main)
