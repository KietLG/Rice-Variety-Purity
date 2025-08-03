import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import gabor

def extract_gist_features(image):
    """
    Trích xuất 128 đặc trưng GIST dựa trên bộ lọc Gabor:
      - Resize ảnh về 128x128, chia thành lưới 4x4,
      - Với 8 hướng lọc, tính trung bình độ lớn ở mỗi ô.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(gray, (128, 128))
    num_orientations = 8
    frequency = 0.2
    gist_features = []
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = image_resized.shape[0] // grid_rows, image_resized.shape[1] // grid_cols

    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        filt_real, filt_imag = gabor(image_resized, frequency=frequency, theta=theta)
        magnitude = np.sqrt(filt_real**2 + filt_imag**2)
        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = magnitude[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                gist_features.append(np.mean(cell))
    return gist_features

def extract_all_features(image):
    """Chỉ lấy đặc trưng GIST."""
    return extract_gist_features(image)

def process_images(folder_path, label):
    """Duyệt qua các ảnh trong folder và trích xuất vector đặc trưng kèm nhãn."""
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            feats = extract_all_features(image)
            features_list.append([label] + feats)
    return features_list

def extract_and_save_gist_features(folder_positive, folder_negative, output_csv):
    """Trích xuất đặc trưng GIST từ hai thư mục và lưu vào CSV."""
    features_positive = process_images(folder_positive, os.path.basename(folder_positive))
    features_negative = process_images(folder_negative, os.path.basename(folder_negative))
    all_features = features_positive + features_negative

    label_column = ['Label']
    gist_cols = [f'GIST_{i}' for i in range(128)]
    columns = label_column + gist_cols

    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved features of success! (File: {output_csv})")

datasets = [
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\negative",
        'output_csv': 'BC15_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\negative",
        'output_csv': 'huongthom_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\negative",
        'output_csv': 'nep87_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\negative",
        'output_csv': 'q5_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\negative",
        'output_csv': 'tbr36_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\negative",
        'output_csv': 'tbr45_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\negative",
        'output_csv': 'th35_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\negative",
        'output_csv': 'thienuu_gist.csv'
    },
    {
        'folder_positive': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\positive",
        'folder_negative': r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\negative",
        'output_csv': 'xi23_gist.csv'
    }
]

for dataset in datasets:
    extract_and_save_gist_features(dataset['folder_positive'], dataset['folder_negative'], dataset['output_csv'])