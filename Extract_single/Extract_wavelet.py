import os
import cv2
import numpy as np
import pandas as pd
import pywt
def extract_wavelet_features(image, wavelet='db1'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(gray, wavelet)
    LL, (LH, HL, HH) = coeffs
    features = []
    for subband in [LL, LH, HL, HH]:
        features.extend([np.mean(subband), np.std(subband), np.sum(np.square(subband))])
    return features
def extract_all_features(image):
    return extract_wavelet_features(image, wavelet='db1')

def process_images(folder_path, label):
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
def process_dataset(folder_positive, folder_negative, output_csv):
    label_pos = os.path.basename(folder_positive)
    label_neg = os.path.basename(folder_negative)
    
    features_pos = process_images(folder_positive, label_pos)
    features_neg = process_images(folder_negative, label_neg)
    all_features = features_pos + features_neg
    label_column = ['Label']
    wavelet_cols = []
    for sb in ['LL', 'LH', 'HL', 'HH']:
        for stat in ['mean', 'std', 'energy']:
            wavelet_cols.append(f'Wavelet_{sb}_{stat}')
    columns = label_column + wavelet_cols

    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved features of success! (File: {output_csv})")
if __name__ == '__main__':
    datasets = [
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\negative",
            "output_csv": "BC15_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\negative",
            "output_csv": "Huongthom_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\negative",
            "output_csv": "Nep87_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\negative",
            "output_csv": "Q5_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\negative",
            "output_csv": "TBR36_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\negative",
            "output_csv": "TBR45_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\negative",
            "output_csv": "TH35_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\negative",
            "output_csv": "Thienuu_wavelet.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\negative",
            "output_csv": "Xi23_wavelet.csv"
        }
    ]

    for dataset in datasets:
        process_dataset(dataset["folder_positive"],
                        dataset["folder_negative"],
                        dataset["output_csv"])
