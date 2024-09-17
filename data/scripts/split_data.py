import os
import shutil
import random

def split_data(src_folder, dest_folder, split_ratio=0.8):
    categories = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'very mild Dementia']
    for category in categories:
        src_dir = os.path.join(src_folder, category)
        files = os.listdir(src_dir)
        random.shuffle(files)
        
        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]
        
        train_dir = os.path.join(dest_folder, 'train', category)
        val_dir = os.path.join(dest_folder, 'val', category)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        for file in train_files:
            shutil.copy(os.path.join(src_dir, file), os.path.join(train_dir, file))
        
        for file in val_files:
            shutil.copy(os.path.join(src_dir, file), os.path.join(val_dir, file))

src_folder = "C:\\Users\\golla\\Downloads\\archive\\Data"
dest_folder = 'C:\\Users\\golla\\OneDrive\\Desktop\\NeuroNet\\data'
split_data(src_folder, dest_folder)
