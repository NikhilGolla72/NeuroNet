import os
import shutil

def split_data(src_dir, dest_dir):
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate over each subdirectory (e.g., 'Mild Dementia', 'Moderate Dementia')
    for category in os.listdir(src_dir):
        category_path = os.path.join(src_dir, category)

        # Ensure the path is a directory
        if os.path.isdir(category_path):
            # Get all files in the category folder
            files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            train_files = files[:int(0.8 * len(files))]
            val_files = files[int(0.8 * len(files)):]

            # Create subfolders in train and val directories
            train_subdir = os.path.join(train_dir, category)
            val_subdir = os.path.join(val_dir, category)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(val_subdir, exist_ok=True)

            # Copy train files
            for file in train_files:
                src_file = os.path.join(category_path, file)
                dst_file = os.path.join(train_subdir, file)
                shutil.copy(src_file, dst_file)

            # Copy val files
            for file in val_files:
                src_file = os.path.join(category_path, file)
                dst_file = os.path.join(val_subdir, file)
                shutil.copy(src_file, dst_file)

src_folder = "/Users/bhanuprakash/Desktop/Neuro/NeuroNet/data/train"
dest_folder = "/Users/bhanuprakash/Desktop/Neuro/NeuroNet/data/split"  # Make sure this is a different folder

split_data(src_folder, dest_folder)