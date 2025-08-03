import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label

def calc_skew(channel):
    """Tính skewness cho một mảng numpy (channel)."""
    c_mean = np.mean(channel)
    c_std = np.std(channel) + 1e-5
    return np.mean((channel - c_mean) ** 3) / (c_std ** 3)

def calc_kurtosis(channel):
    """Tính kurtosis cho một mảng numpy (channel)."""
    c_mean = np.mean(channel)
    c_std = np.std(channel) + 1e-5
    return np.mean((channel - c_mean) ** 4) / (c_std ** 4) - 3

def channel_stats(channel):
    """Trả về: mean, std, skewness, kurtosis, median, min, max của channel."""
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
    """
    Trích xuất 21 đặc trưng từ không gian HSV:
      Với mỗi kênh (H, S, V) tính: mean, std, skewness, kurtosis, median, min, max.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv)
    features = []
    for ch in channels:
        features.extend(channel_stats(ch))
    return features

def extract_basic_features(image):
    """
    Trích xuất 30 đặc trưng nâng cao gồm:
      - 14 đặc trưng hình học của đối tượng (dựa trên vùng được xác định bằng Otsu thresholding).
      - 12 đặc trưng màu RGB: với mỗi kênh tính mean, sqrt(mean), std, skewness.
      - 4 đặc trưng texture từ vùng đối tượng: mean, std, uniformity, third moment.
    """
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
        orientation = getattr(props, 'orientation', 0)
        solidity = getattr(props, 'solidity', 0)
        extent = getattr(props, 'extent', 0)
        roundness = (4.0 * np.pi * area) / ((perimeter**2) + epsilon) if perimeter > 0 else 0

        region_pixels = gray[props.coords[:, 0], props.coords[:, 1]].astype(np.float32)
        if region_pixels.size > 0:
            texture_mean = np.mean(region_pixels)
            texture_std = np.std(region_pixels)
            hist, _ = np.histogram(region_pixels, bins=256, range=(0,256), density=True)
            texture_uniformity = np.sum(hist**2)
            texture_third_moment = np.mean((region_pixels - texture_mean)**3)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_rgb = np.mean(image_rgb, axis=(0, 1))
    sqrt_mean_rgb = np.sqrt(mean_rgb)
    std_rgb = np.std(image_rgb, axis=(0, 1))
    skew_rgb = [calc_skew(image_rgb[:, :, i]) for i in range(3)]

    enhanced_features = [
        area, length, width_obj, length_width_ratio,
        major_axis, minor_axis, convex_area, convex_perimeter,
        perimeter, eccentricity, orientation, solidity, extent, roundness,
        mean_rgb[0], mean_rgb[1], mean_rgb[2],
        sqrt_mean_rgb[0], sqrt_mean_rgb[1], sqrt_mean_rgb[2],
        std_rgb[0], std_rgb[1], std_rgb[2],
        skew_rgb[0], skew_rgb[1], skew_rgb[2],
        texture_mean, texture_std, texture_uniformity, texture_third_moment
    ]
    return enhanced_features

def extract_all_features(image):
    """Ghép nối đặc trưng basic và HSV thành một vector đặc trưng."""
    feat_basic = extract_basic_features(image)   
    feat_hsv = extract_hsv_features(image)       
    return feat_basic + feat_hsv

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

label_column = ['Label']
enhanced_cols = [
    'Area', 'Length', 'Width', 'LengthWidthRatio',
    'MajorAxisLength', 'MinorAxisLength', 'ConvexArea', 'ConvexPerimeter',
    'Perimeter', 'Eccentricity', 'Orientation', 'Solidity', 'Extent', 'Roundness',
    'Mean_R', 'Mean_G', 'Mean_B',
    'SqrtMean_R', 'SqrtMean_G', 'SqrtMean_B',
    'Std_R', 'Std_G', 'Std_B',
    'Skew_R', 'Skew_G', 'Skew_B',
    'Texture_Mean', 'Texture_Std', 'Texture_Uniformity', 'Texture_ThirdMoment'
]
hsv_channels = ['H', 'S', 'V']
hsv_stats = ['mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max']
hsv_cols = [f'HSV_{ch}_{stat}' for ch in hsv_channels for stat in hsv_stats]
columns = label_column + enhanced_cols + hsv_cols

datasets = [
    ('BC-15', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\BC-15\negative", 'BC15_basic_hsv.csv'),
    ('Huong_thom-1', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Huong_thom-1\negative", 'huongthom_basic_hsv.csv'),
    ('Nep-87', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Nep-87\negative", 'nep_basic_hsv.csv'),
    ('Q-5', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Q-5\negative", 'q5_basic_hsv.csv'),
    ('TBR-36', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-36\negative", 'tbr36_basic_hsv.csv'),
    ('TBR-45', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TBR-45\negative", 'tbr45_basic_hsv.csv'),
    ('TH3-5', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\TH3-5\negative", 'th35_basic_hsv.csv'),
    ('Thien_uu-8', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Thien_uu-8\negative", 'thienuu_basic_hsv.csv'),
    ('Xi-23', r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\positive", 
     r"D:\FPT University\Code_FPT\Season 4\AIL303m\gao\Datasets2\Data\Xi-23\negative", 'xi_basic_hsv.csv')
]

""" Trích xuất đặc trưng và lưu vào file CSV """
for dataset_name, pos_folder, neg_folder, output_file in datasets:
    print(f"Processing dataset: {dataset_name}")
    features1 = process_images(pos_folder, os.path.basename(pos_folder))
    features2 = process_images(neg_folder, os.path.basename(neg_folder))
    all_features = features1 + features2
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Saved features of success! (File: {dataset_name})")