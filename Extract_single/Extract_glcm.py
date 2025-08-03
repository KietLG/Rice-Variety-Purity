import os
import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
def extract_glcm_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        vals = graycoprops(glcm, prop)
        features.extend(vals[0, :].tolist())
    dissim_vals = graycoprops(glcm, 'dissimilarity')[0, :].tolist()
    asm_vals = (graycoprops(glcm, 'energy')**2)[0, :].tolist()
    features.extend(dissim_vals)
    features.extend(asm_vals)
    return features
def extract_all_features(image):
    return extract_glcm_features(image)

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
    basic_props = ['contrast', 'correlation', 'energy', 'homogeneity']
    glcm_cols = [f'GLCM_{p}_{a}' for p in basic_props for a in ['0', '45', '90', '135']]
    additional_props = ['dissimilarity', 'ASM']
    glcm_cols += [f'GLCM_{p}_{a}' for p in additional_props for a in ['0', '45', '90', '135']]
    
    columns = label_column + glcm_cols

    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved features of success! (File: {output_csv})")
if __name__ == '__main__':
    datasets = [
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\negative",
            "output_csv": "BC15_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\negative",
            "output_csv": "Huongthom_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\negative",
            "output_csv": "Nep87_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\negative",
            "output_csv": "Q5_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\negative",
            "output_csv": "TBR36_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\negative",
            "output_csv": "TBR45_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\negative",
            "output_csv": "TH35_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\negative",
            "output_csv": "Thienuu_glcm.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\negative",
            "output_csv": "Xi23_glcm.csv"
        }
    ]

    for dataset in datasets:
        process_dataset(dataset["folder_positive"],
                        dataset["folder_negative"],
                        dataset["output_csv"])
