# Rice Variety Purity Classification Using Computer Vision

## Overview

This project implements an automated system for classifying the purity of rice seed varieties using image processing and machine learning techniques, as described in the research paper *"Combining Visual Descriptors for Accurate Rice Variety Purity Classification"* by Thi-Thu-Hong Phan et al. The system aims to replace labor-intensive and error-prone manual inspection methods by leveraging advanced computer vision and machine learning approaches. By integrating multiple visual descriptors—such as GLCM, GIST, LBP, Basic, Hu, Wavelet, and Color features—along with machine learning models including KNN, SVM, Random Forest, and CatBoost, the system achieves a remarkable accuracy of **96.65%** for the Thien-un-8 rice variety on a real-world Vietnamese rice seed dataset.

This project supports agricultural production by ensuring seed purity, enhancing crop yield, and promoting precision agriculture practices.

## Key Features

- **Diverse Feature Extraction**: Utilizes a variety of visual descriptors:
  - **Morphological Features**: Includes area, length, width, length-width ratio, major/minor axis lengths, convexity, perimeter, eccentricity, orientation, solidity, extent, and roundness.
  - **Color Features**: Statistical measures across RGB, HSV, HLS, LAB, and YCbCr color spaces, including mean, standard deviation, skewness, kurtosis, median, minimum, and maximum values.
  - **Texture Features**: Employs GLCM, LBP, and Wavelet transforms to capture texture patterns.
  - **GIST Descriptor**: Extracts global image characteristics for holistic representation.
  - **Hu Moments**: Provides rotation and scale-invariant features.
- **Feature Fusion**: Combines multiple descriptors (295 features in total) to create a unified representation, enhancing classification accuracy.
- **Machine Learning Models**: Evaluates KNN, SVM, Random Forest, Logistic Regression, XGBoost, CatBoost, and Extra Trees for robust classification.
- **Real-World Dataset**: Uses images of nine rice varieties from Northern Vietnam (BC-15, Huong-thom-1, Nep-87, Q-5, TBR-36, TBR-45, TH3-5, Thien-un-8, Xi-23).

## Installation

### Prerequisites

- Python 3.11 or higher
- Required Python libraries:
  - `numpy`
  - `opencv-python`
  - `scikit-learn`
  - `catboost`
  - `xgboost`
  - `pandas`
  - `matplotlib`
  - `scikit-image`

### Setup Instructions

1. **Install Python**: Ensure Python 3.11+ is installed. Download from python.org.
2. **Install Dependencies**: pip install numpy opencv-python scikit-learn catboost xgboost pandas matplotlib scikit-image
3. **Clone the Repository**: https://github.com/KietLG/Rice-Variety-Purity.git

## Dataset

The dataset consists of images from nine widely cultivated rice varieties in Northern Vietnam:

- **Rice Varieties**: BC-15, Huong-thom-1, Nep-87, Q-5, TBR-36, TBR-45, TH3-5, Thien-un-8, Xi-23.
- **Positive Samples**: Verified images representing each specific variety.
- **Negative Samples**: Images of rice seeds not belonging to the nine varieties, simulating real-world classification scenarios.
- **Balanced Dataset**: Equal representation of each variety to ensure fair training and testing.

**Note**: Due to copyright restrictions, the dataset is not included in this repository. Contact the paper's authors for access.

## Results

The system demonstrates superior performance when combining visual descriptors with the CatBoost model:

- **Highest Accuracy**: 96.65% for the Thien-un-8 variety.
- **Average Performance**:
  - GLCM Features: 80.11% (CatBoost).
  - LBP Features: 68.11% (SVM).
  - Color Features: 95.55% (CatBoost).
- **Observations**:
  - Nep-87 was the easiest variety to classify.
  - BC-15 was the most challenging across all models.
  - Feature fusion significantly improves accuracy compared to using individual descriptors.

## Contributing

We welcome contributions from the community! To contribute:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make changes and commit: `git commit -m "Describe your changes"`.
4. Push to your branch: `git push origin feature/your-feature-name`.
5. Submit a Pull Request on GitHub.

Please ensure code adheres to PEP 8 standards and includes detailed documentation.

## License

This project is licensed under the MIT License.

## References

- **Original Paper**: Thi-Thu-Hong Phan et al., *"Combining Visual Descriptors for Accurate Rice Variety Purity Classification"*, FPT University, Da Nang, Vietnam.
- **Contact**: Email: lamgiakiet16032004@gmail.com or tranthigam27072004@gmail.com
- **Issues**: Report bugs or ask questions via GitHub Issues.