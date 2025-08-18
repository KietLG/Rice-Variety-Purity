import os
import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import regionprops, label
from skimage.filters import gabor

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

def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return [-1 * np.sign(h) * np.log10(abs(h) + 1e-15) for h in hu]

def extract_wavelet_features(image, wavelet='db1'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(gray, wavelet)
    LL, (LH, HL, HH) = coeffs
    features = []
    for subband in [LL, LH, HL, HH]:
        features.extend([np.mean(subband), np.std(subband), np.sum(np.square(subband))])
    return features

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

def extract_basic_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled = label(binary)
    regions = regionprops(labeled)
    epsilon = 1e-5

    (area, length, width_obj, length_width_ratio, major_axis, minor_axis,
     convex_area, convex_perimeter, perimeter, eccentricity, orientation,
     solidity, extent, roundness) = [0]*14
    texture_mean = texture_std = texture_uniformity = texture_third_moment = 0

    if regions:
        props = max(regions, key=lambda r: r.area)
        area = props.area
        minr, minc, maxr, maxc = props.bbox
        bbox_height, bbox_width = maxr - minr, maxc - minc
        length = max(bbox_height, bbox_width)
        width_obj = min(bbox_height, bbox_width)
        length_width_ratio = length / (width_obj + epsilon)
        major_axis = getattr(props, 'major_axis_length', 0)
        minor_axis = getattr(props, 'minor_axis_length', 0)
        convex_area = getattr(props, 'convex_area', 0)
        pts = np.array([[c[1], c[0]] for c in props.coords], dtype=np.int32)
        hull = cv2.convexHull(pts)
        convex_perimeter = cv2.arcLength(hull, True)
        perimeter = getattr(props, 'perimeter', 0)
        eccentricity = getattr(props, 'eccentricity', 0)
        orientation  = getattr(props, 'orientation', 0)
        solidity     = getattr(props, 'solidity', 0)
        extent       = getattr(props, 'extent', 0)
        roundness    = (4.0 * np.pi * area) / ((perimeter**2) + epsilon) if perimeter > 0 else 0

        region_pixels = gray[props.coords[:, 0], props.coords[:, 1]].astype(np.float32)
        if region_pixels.size > 0:
            texture_mean = np.mean(region_pixels)
            texture_std  = np.std(region_pixels)
            hist, _ = np.histogram(region_pixels, bins=256, range=(0,256), density=True)
            texture_uniformity = np.sum(hist**2)
            texture_third_moment = np.mean((region_pixels - texture_mean)**3)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_rgb = np.mean(image_rgb, axis=(0, 1))
    sqrt_mean_rgb = np.sqrt(mean_rgb)
    std_rgb = np.std(image_rgb, axis=(0, 1))
    skew_rgb = [calc_skew(image_rgb[:, :, i]) for i in range(3)]

    basic_features = [
        area, length, width_obj, length_width_ratio,
        major_axis, minor_axis, convex_area, convex_perimeter,
        perimeter, eccentricity, orientation, solidity, extent, roundness,
        mean_rgb[0], mean_rgb[1], mean_rgb[2],
        sqrt_mean_rgb[0], sqrt_mean_rgb[1], sqrt_mean_rgb[2],
        std_rgb[0], std_rgb[1], std_rgb[2],
        skew_rgb[0], skew_rgb[1], skew_rgb[2],
        texture_mean, texture_std, texture_uniformity, texture_third_moment
    ]
    return basic_features

def extract_gist_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(gray, (128, 128))
    num_orientations = 8
    frequency = 0.2
    gist_features = []
    grid_rows, grid_cols = 4, 4
    cell_h = image_resized.shape[0] // grid_rows
    cell_w = image_resized.shape[1] // grid_cols

    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        filt_real, filt_imag = gabor(image_resized, frequency=frequency, theta=theta)
        magnitude = np.sqrt(filt_real**2 + filt_imag**2)
        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = magnitude[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                gist_features.append(np.mean(cell))
    return gist_features

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

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-5)
    return hist.tolist()

def extract_all_features(image):
    feat_gist = extract_gist_features(image)    # 128
    feat_glcm = extract_glcm_features(image)      # 24
    feat_lbp = extract_lbp_features(image)       # 10
    feat_basic = extract_basic_features(image)     # 30
    feat_hu = extract_hu_moments(image)         # 7
    feat_wavelet = extract_wavelet_features(image, wavelet='db1')  # 12
    feat_hsv = extract_hsv_features(image)       # 21
    feat_hls = extract_hls_features(image)       # 21
    feat_lab = extract_lab_features(image)       # 21
    feat_ycrcb = extract_ycrcb_features(image)     # 21

    return (feat_gist + feat_glcm + feat_lbp + feat_basic + feat_hu + feat_wavelet + feat_hsv + feat_hls + feat_lab + feat_ycrcb)

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
    gist_cols = [f'GIST_{i}' for i in range(128)]
    basic_props = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']
    glcm_cols = [f'GLCM_{p}_{a}' for p in basic_props for a in ['0', '45', '90', '135']]
    additional_props = ['Dissimilarity', 'ASM']
    glcm_cols += [f'GLCM_{p}_{a}' for p in additional_props for a in ['0', '45', '90', '135']]
    lbp_cols = [f'LBP_{i}' for i in range(10)]
    basic_cols = ['Area', 'Length', 'Width', 'LengthWidthRatio', 'MajorAxisLength', 'MinorAxisLength', 'ConvexArea', 'ConvexPerimeter', 'Perimeter', 'Eccentricity', 'Orientation', 'Solidity', 'Extent', 'Roundness', 
                  'Mean_R', 'Mean_G', 'Mean_B', 'SqrtMean_R', 'SqrtMean_G', 'SqrtMean_B', 'Std_R', 'Std_G', 'Std_B', 'Skew_R', 'Skew_G', 'Skew_B',
                  'Texture_Mean', 'Texture_Std', 'Texture_Uniformity', 'Texture_ThirdMoment'
    ]
    hu_cols = [f'Hu_{i+1}' for i in range(7)]
    wavelet_cols = []
    for sb in ['LL', 'LH', 'HL', 'HH']:
        for stat in ['mean', 'std', 'energy']:
            wavelet_cols.append(f'Wavelet_{sb}_{stat}')
    hsv_channels = ['H', 'S', 'V']
    hsv_stats = ['mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max']
    hsv_cols = [f'HSV_{ch}_{stat}' for ch in hsv_channels for stat in hsv_stats]
    hls_channels = ['H', 'L', 'S']
    hls_cols = [f'HLS_{ch}_{stat}' for ch in hls_channels for stat in hsv_stats]
    lab_channels = ['L', 'A', 'B']
    lab_cols = [f'LAB_{ch}_{stat}' for ch in lab_channels for stat in hsv_stats]
    ycrcb_channels = ['Y', 'Cr', 'Cb']
    ycrcb_cols = [f'YCrCb_{ch}_{stat}' for ch in ycrcb_channels for stat in hsv_stats]

    columns = (label_column + gist_cols + glcm_cols + lbp_cols + basic_cols +
               hu_cols + wavelet_cols + hsv_cols + hls_cols + lab_cols + ycrcb_cols)

    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved features of success! (File: {output_csv})")

if __name__ == '__main__':
    datasets = [
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\negative",
            "output_csv": "BC15.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\negative",
            "output_csv": "Huongthom.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\negative",
            "output_csv": "Nep87.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\negative",
            "output_csv": "Q5.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\negative",
            "output_csv": "TBR36.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\negative",
            "output_csv": "TBR45.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\negative",
            "output_csv": "TH35.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\negative",
            "output_csv": "Thienuu.csv"
        },
        {
            "folder_positive": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\positive",
            "folder_negative": r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\negative",
            "output_csv": "Xi23.csv"
        }
    ]

    for dataset in datasets:
        process_dataset(dataset["folder_positive"],
                        dataset["folder_negative"],
                        dataset["output_csv"])