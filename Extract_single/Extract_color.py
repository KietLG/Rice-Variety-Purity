import os
import cv2
import numpy as np
import pandas as pd

def calc_skew(channel):
    c_mean = np.mean(channel)
    c_std = np.std(channel) + 1e-5
    return np.mean((channel - c_mean) ** 3) / (c_std ** 3)

def calc_kurtosis(channel):
    c_mean = np.mean(channel)
    c_std = np.std(channel) + 1e-5
    return np.mean((channel - c_mean) ** 4) / (c_std ** 4) - 3

def channel_stats(channel):
    return [
        np.mean(channel),
        np.std(channel),
        calc_skew(channel),
        calc_kurtosis(channel),
        np.median(channel),
        np.min(channel),
        np.max(channel)
    ]
def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv)
    features = []
    for ch in channels:
        features.extend(channel_stats(ch))
    return features
def extract_hls_features(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    channels = cv2.split(hls)
    features = []
    for ch in channels:
        features.extend(channel_stats(ch))
    return features
def extract_lab_features(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)
    features = []
    for ch in channels:
        features.extend(channel_stats(ch))
    return features
def extract_ycrcb_features(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    features = []
    for ch in channels:
        features.extend(channel_stats(ch))
    return features
def extract_color_features(image):
    feat_hsv = extract_hsv_features(image)     
    feat_hls = extract_hls_features(image)     
    feat_lab = extract_lab_features(image)    
    feat_ycrcb = extract_ycrcb_features(image)  
    return feat_hsv + feat_hls + feat_lab + feat_ycrcb  
def process_images(folder_path, label):
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            feats = extract_color_features(image)
            features_list.append([label] + feats)
    return features_list
def process_dataset(folder_positive, folder_negative, output_csv):
    label_pos = os.path.basename(folder_positive)
    label_neg = os.path.basename(folder_negative)
    
    features_pos = process_images(folder_positive, label_pos)
    features_neg = process_images(folder_negative, label_neg)
    all_features = features_pos + features_neg

    label_column = ['Label']
    color_spaces = {
        'HSV': ['H', 'S', 'V'],
        'HLS': ['H', 'L', 'S'],
        'LAB': ['L', 'A', 'B'],
        'YCrCb': ['Y', 'Cr', 'Cb']
    }
    stats = ['mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max']
    columns = label_column
    for space, channels in color_spaces.items():
        for ch in channels:
            for stat in stats:
                columns.append(f'{space}_{ch}_{stat}')
    
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved features of success! (File: {output_csv})")

datasets = [
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\negative",
        "output_csv": "BC15_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\negative",
        "output_csv": "Huongthom_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\negative",
        "output_csv": "Nep87_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\negative",
        "output_csv": "Q5_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\negative",
        "output_csv": "TBR36_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\negative",
        "output_csv": "TBR45_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\negative",
        "output_csv": "TH35_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\negative",
        "output_csv": "Thienuu_color.csv"
    },
    {
        "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\positive",
        "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\negative",
        "output_csv": "Xi23_color.csv"
    }
]

for dataset in datasets:
    process_dataset(dataset["folder_positive"],
                    dataset["folder_negative"],
                    dataset["output_csv"])
