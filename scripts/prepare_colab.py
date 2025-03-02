"""
This script prepares your local development code for Colab training.
It creates a zip file containing only the necessary files and prints
instructions for uploading to Colab.
"""
import os
import shutil
import datetime

def prepare_for_colab():
    # Create a temporary directory for Colab files
    temp_dir = f"colab_upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define files and directories to copy
    copy_items = [
        'src',
        'configs',
        'scripts',
        'requirements.txt',
        'notebooks/train_drone_nav.ipynb'
    ]
    
    # Copy files
    for item in copy_items:
        if os.path.isdir(item):
            shutil.copytree(item, os.path.join(temp_dir, item))
        else:
            shutil.copy2(item, temp_dir)
    
    # Create zip file
    shutil.make_archive(temp_dir, 'zip', temp_dir)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    print("\nColab preparation complete!")
    print(f"Created: {temp_dir}.zip")
    print("\nInstructions for Colab training:")
    print("1. Upload the zip file to your Google Drive")
    print("2. Open the notebook in Colab")
    print("3. Mount your Google Drive and unzip the files")
    print("4. Follow the training instructions in the notebook")

if __name__ == "__main__":
    prepare_for_colab()