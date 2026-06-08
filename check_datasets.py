# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

# china-phenology
base = 'D:/crop_datasets/raw/china-phenology/Six Crops'
print('=== china-phenology/Six Crops ===')
for crop in sorted(os.listdir(base)):
    crop_path = os.path.join(base, crop)
    if os.path.isdir(crop_path):
        print(f'\n{crop}:')
        for stage in sorted(os.listdir(crop_path)):
            stage_path = os.path.join(crop_path, stage)
            if os.path.isdir(stage_path):
                count = len([f for f in os.listdir(stage_path) if os.path.isfile(os.path.join(stage_path, f))])
                print(f'  {stage}: {count} files')

# 生长周期
print('\n\n=== 生长周期 ===')
base2 = 'D:/crop_datasets/raw/生长周期/生长周期'
for crop in sorted(os.listdir(base2)):
    crop_path = os.path.join(base2, crop)
    if os.path.isdir(crop_path):
        print(f'\n{crop}:')
        for stage in sorted(os.listdir(crop_path)):
            stage_path = os.path.join(crop_path, stage)
            if os.path.isdir(stage_path):
                count = len([f for f in os.listdir(stage_path) if os.path.isfile(os.path.join(stage_path, f))])
                print(f'  {stage}: {count} files')
