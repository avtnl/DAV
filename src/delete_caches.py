import os
import shutil

def delete_pycache(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                shutil.rmtree(full_path)
                print(f"Deleted: {full_path}")
        for filename in filenames:
            if filename.endswith(".pyc"):
                full_path = os.path.join(dirpath, filename)
                os.remove(full_path)
                print(f"Deleted: {full_path}")

delete_pycache("src")  # or "." for full project