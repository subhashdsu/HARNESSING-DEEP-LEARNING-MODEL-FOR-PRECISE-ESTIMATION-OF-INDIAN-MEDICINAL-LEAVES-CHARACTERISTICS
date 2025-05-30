import os
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
base_dir = 'dataset2'
target_dir = 'augmenteddataset2'
number_of_combinations = 2  # User can set this value

# Create target directory
os.makedirs(target_dir, exist_ok=True)

# Initialize ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Loop over subdirectories
for subdir in tqdm(os.listdir(base_dir), desc="Processing images", unit="dir"):
    # Get list of files in subdir
    subdir_path = os.path.join(base_dir, subdir)
    files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
    
    # Shuffle files
    np.random.shuffle(files)
    
    # Split files into original and augmented sets
    original_files = files[:len(files)//2]  # original images
    augmented_files = files[len(files)//2:]  # augmented images
    
    # Create class directories in target directory
    os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)
    
    # Copy original files to target directory
    for f in original_files:
        shutil.copy(f, os.path.join(target_dir, subdir))
    
    # Augment and copy augmented files to target directory
    for f in augmented_files:
        image = tf.keras.preprocessing.image.load_img(f)
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(target_dir, subdir), save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= number_of_combinations:  # create specified number of augmented images per original image
                break

print("Dataset augmented and saved in target directory.")
