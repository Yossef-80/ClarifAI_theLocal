import os
import shutil

# === CONFIGURATION ===
folder_path = r"D:\ClarifAI_web\clarifAI-backend"       # Folder where the file should be replaced
file_to_remove = "schooling_system.db"                 # Name of the file to remove
file_to_copy = r"D:\ClarifAI_web\schooling_system.db"        # Full path of the file to copy

# === REMOVE OLD FILE ===
target_file_path = os.path.join(folder_path, file_to_remove)
if os.path.exists(target_file_path):
    os.remove(target_file_path)
    print(f"Removed: {target_file_path}")
else:
    print(f"No existing file to remove: {target_file_path}")

# === COPY NEW FILE ===
try:
    shutil.copy(file_to_copy, folder_path)
    print(f"Copied {file_to_copy} → {folder_path}")
except FileNotFoundError:
    print(f"Error: File to copy not found: {file_to_copy}")
except Exception as e:
    print(f"Error while copying file: {e}")
