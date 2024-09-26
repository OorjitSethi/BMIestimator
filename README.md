# BMI Estimator from Facial Features

## Project Overview
This project is designed to estimate a person's Body Mass Index (BMI) using facial features extracted from an image. It leverages machine learning and image processing techniques to predict BMI based on specific facial metrics, such as the width-to-height ratio and cheek-to-jaw width ratio. The model was trained on a dataset of 526 entries, and although it shows promise, the results may not be perfectly accurate due to the limited size of the dataset and the inherent complexity of estimating BMI from facial features alone.

## Dataset Description
The dataset used for training the model contains 526 data points. Each data point includes various attributes such as:

- **ID**: Unique identifier for each entry.
- **age**: Age of the individual.
- **bmi**: Body Mass Index (target variable).
- **height**: Height of the individual in centimeters.
- **weight**: Weight of the individual in kilograms.
- **WtoHRatio**: Width-to-Height Ratio of the face.
- **CheektoJawWidth**: Ratio of cheek width to jaw width.
- **ParameterAreaRatio**: A calculated ratio indicative of facial area.
- **shape.PC1 - shape.PC65**: Principal components representing shape features of the face.
- **color.PC1 - color.PC286**: Principal components representing color features of the face.

The columns `shape.PC1` to `shape.PC65` capture the variance in facial shape, while `color.PC1` to `color.PC286` capture the variance in facial color.

### Sample Dataset Structure
| ID | age | bmi  | height | weight | WtoHRatio | CheektoJawWidth | ParameterAreaRatio | shape.PC1 | ... | color.PC286 |
|----|-----|------|--------|--------|-----------|-----------------|--------------------|-----------|-----|-------------|
| 1  | 20.5| 22.9 | 173.5  | 69     | 2.041     | 1.233           | 0.009              | -11.52    | ... | 0.149       |

## Technical Approach

### 1. Data Preprocessing
- **Data Cleaning**: Rows with missing values are dropped to ensure data integrity.

- **Feature Scaling**:
  - `StandardScaler` is used to scale the feature columns (shape.PC and color.PC components) to zero mean and unit variance.
  - The target variable (`bmi`) is also scaled to improve model performance.

- **Feature Selection**:
  - Correlation analysis is conducted to identify features that are highly correlated with BMI.
  - Features with absolute correlation values greater than 0.1 are selected for training the model.

### 2. Model Training
A `RandomForestRegressor` is used for predicting BMI based on the selected features. The model is trained using a portion of the dataset, with the following steps:

- **Train-Test Split**: The data is split into training and testing sets using an 80-20 ratio.
- **Grid Search for Hyperparameter Tuning**: A grid search is performed over hyperparameters like `n_estimators`, `max_depth`, and `min_samples_split` to find the best model.

### 3. Image Processing
- **Facial Landmark Detection**: Dlib's pre-trained `shape_predictor_68_face_landmarks.dat` model is used to detect 68 facial landmarks from the input image.
- **Metric Computation**: Using the detected landmarks, key facial metrics such as width-to-height ratio and cheek-to-jaw width ratio are computed.

### 4. BMI Prediction
- **Feature Vector Preparation**: The computed facial metrics are integrated into a feature vector, which is then scaled using the same scaler as the training data.
- **Prediction**: The scaled feature vector is fed into the trained `RandomForestRegressor` model to predict BMI. The predicted BMI is then transformed back to the original scale.

## Limitations and Uncertainties
- **Dataset Size**: The model is trained on a limited dataset of 526 samples. This small sample size may not capture the full variance in human facial features and their relationship with BMI.
- **Prediction Uncertainty**: The model shows an absolute uncertainty of approximately ±0.5 in terms of BMI during testing. However, this uncertainty could be higher for unseen data.
- **Generalization**: The model's predictions may not generalize well to all population groups due to potential biases in the dataset.

> **Disclaimer**: This tool is not a substitute for professional medical advice. For any health-related concerns, please consult a healthcare professional. The BMI predictions made by this model should be considered as rough estimates and not definitive indicators of health.

## Conclusion
The BMI Estimator project showcases the application of machine learning and image processing techniques to predict BMI based on facial features. While it demonstrates the potential of such an approach, further refinement and a larger dataset are needed to improve accuracy and generalizability.
