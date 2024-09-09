import cv2
import os
import random


def process_images(input_file, srcpath, tarpath):
    output_file = 'trainset_label_sample.txt'
    num_samples = 100000
    kernels = 3
    suffix = '.jpg'

    with open(input_file, 'r') as file:
        lines = file.readlines()

    data_lines = lines[1:]
    selected_lines = random.sample(data_lines, num_samples)
    processed_lines = []  # Initialize with header

    for line in selected_lines:
        img_name, label = line.strip().split(',')
        img_path = os.path.join(srcpath, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        mean_path = os.path.join(tarpath, img_name.replace(suffix, '_mean' + suffix))
        gaussian_path = os.path.join(tarpath, img_name.replace(suffix, '_gaussian' + suffix))
        median_path = os.path.join(tarpath, img_name.replace(suffix, '_median' + suffix))
        os.makedirs(os.path.dirname(mean_path), exist_ok=True)
        os.makedirs(os.path.dirname(gaussian_path), exist_ok=True)
        os.makedirs(os.path.dirname(median_path), exist_ok=True)
        
        # meanBlur
        img_mean = cv2.blur(img, (kernels, kernels))
        cv2.imwrite(mean_path, img_mean)
        
        # GaussianBlur
        img_gaussian = cv2.GaussianBlur(img, (kernels, kernels), 0)
        cv2.imwrite(gaussian_path, img_gaussian)
        
        # medianBlur
        img_median = cv2.medianBlur(img, kernels)
        cv2.imwrite(median_path, img_median)
        
        processed_lines.append(f"{img_name.replace(suffix, '_mean' + suffix)}, {label}\n")
        processed_lines.append(f"{img_name.replace(suffix, '_gaussian' + suffix)}, {label}\n")
        processed_lines.append(f"{img_name.replace(suffix, '_median' + suffix)}, {label}\n")

    with open(output_file, 'w') as file:
        file.writelines(processed_lines)
    
    return output_file

def append_labels(src_file, dest_file):
    with open(src_file, 'r') as src:
        content = src.read()  
    with open(dest_file, 'a') as dest:
        dest.write(content)  
