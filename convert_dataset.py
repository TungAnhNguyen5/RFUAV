#!/usr/bin/env python3
"""
Convert raw IQ data to spectrograms using direct generate_images function.
"""
import os
import shutil
from pathlib import Path
from graphic.RawDataProcessor import generate_images

repo_root = '/home/dev/Documents/RFUAV'
dataset_base = os.path.join(repo_root, 'dataset')

drone_classes = {
    'YUNZHUO H12': 'YUNZHUO_H12',
    'YUNZHUO H16': 'YUNZHUO_H16',
    'YUNZHUO H30': 'YUNZHUO_H30',
}

SAMPLE_RATE = 100e6
STFT_POINT = 1024
DURATION_TIME = 0.1

def main():
    print("=" * 70)
    print("Converting raw IQ data to spectrograms (Direct Method)")
    print("=" * 70)
    
    for drone_folder, class_name in drone_classes.items():
        drone_src = os.path.join(repo_root, drone_folder)
        
        if not os.path.exists(drone_src):
            print(f"✗ Drone folder not found: {drone_src}")
            continue
        
        # Find all .iq files
        iq_files = sorted([f for f in os.listdir(drone_src) if f.endswith('.iq')])
        
        if not iq_files:
            print(f"✗ No .iq files found in {drone_src}")
            continue
        
        print(f"\n📦 Processing {class_name} ({len(iq_files)} files)")
        
        # Split files: first 60% to train, rest to valid
        split_idx = max(1, int(len(iq_files) * 0.6))
        train_files = iq_files[:split_idx]
        val_files = iq_files[split_idx:]
        
        # Convert training files
        train_dir = os.path.join(dataset_base, 'train', class_name, 'imgs')
        os.makedirs(train_dir, exist_ok=True)
        
        for i, iq_file in enumerate(train_files):
            iq_path = os.path.join(drone_src, iq_file)
            print(f"  🔄 Train {i+1}/{len(train_files)}: {iq_file}")
            
            try:
                # Generate images directly
                images = generate_images(
                    datapack=iq_path,
                    fs=int(SAMPLE_RATE),
                    stft_point=STFT_POINT,
                    duration_time=DURATION_TIME,
                    ratio=0,
                    location='buffer'
                )
                
                # Save images
                if images:
                    for img_idx, img in enumerate(images):
                        img_path = os.path.join(train_dir, f"{iq_file}_{img_idx:04d}.png")
                        img.save(img_path)
                    print(f"     ✓ Saved {len(images)} images")
                else:
                    print(f"     ⚠ No images generated")
                    
            except Exception as e:
                print(f"     ✗ Error: {e}")
        
        # Convert validation files
        if val_files:
            val_dir = os.path.join(dataset_base, 'valid', class_name, 'imgs')
            os.makedirs(val_dir, exist_ok=True)
            
            for i, iq_file in enumerate(val_files):
                iq_path = os.path.join(drone_src, iq_file)
                print(f"  🔄 Valid {i+1}/{len(val_files)}: {iq_file}")
                
                try:
                    # Generate images directly
                    images = generate_images(
                        datapack=iq_path,
                        fs=int(SAMPLE_RATE),
                        stft_point=STFT_POINT,
                        duration_time=DURATION_TIME,
                        ratio=0,
                        location='buffer'
                    )
                    
                    # Save images
                    if images:
                        for img_idx, img in enumerate(images):
                            img_path = os.path.join(val_dir, f"{iq_file}_{img_idx:04d}.png")
                            img.save(img_path)
                        print(f"     ✓ Saved {len(images)} images")
                    else:
                        print(f"     ⚠ No images generated")
                        
                except Exception as e:
                    print(f"     ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Conversion complete!")
    print("=" * 70)
    
    # Print dataset summary
    print("\n📊 Dataset Summary:")
    for split in ['train', 'valid']:
        split_path = os.path.join(dataset_base, split)
        print(f"\n{split.upper()}:")
        if os.path.exists(split_path):
            for class_name in sorted(os.listdir(split_path)):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    # Count images in all subdirectories
                    total_imgs = 0
                    for root, dirs, files in os.walk(class_path):
                        total_imgs += len([f for f in files if f.endswith('.png')])
                    print(f"  {class_name:20s}: {total_imgs:5d} images")

if __name__ == '__main__':
    main()
