import pandas as pd
import numpy as np
import cv2
import dlib
from imutils import face_utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('BMI_FM_PCs.txt', sep='\t')

# Drop rows with missing values
data = data.dropna()

# List of feature columns (excluding 'ID', 'age', 'bmi', 'height', 'weight')
excluded_features = ['ID', 'age', 'bmi', 'height', 'weight']
feature_cols = data.columns.drop(excluded_features)

# Initialize the scaler for features
scaler = StandardScaler()

# Fit the scaler on the features
scaler.fit(data[feature_cols])

# Scale the features
data_scaled = data.copy()
data_scaled[feature_cols] = scaler.transform(data[feature_cols])

# Initialize the scaler for BMI
bmi_scaler = StandardScaler()

# Scale the BMI
data_scaled['bmi'] = bmi_scaler.fit_transform(data[['bmi']])

# Compute correlation matrix
corr_matrix = data_scaled.corr()

# Get correlations with 'bmi' only for feature_cols
corr_with_bmi = corr_matrix.loc[feature_cols, 'bmi']

# Select features with high correlation (exclude 'age' and 'weight' if they are present)
high_corr_features = corr_with_bmi[abs(corr_with_bmi) > 0.1].index.tolist()

# Ensure 'age' and 'weight' are not in high_corr_features
high_corr_features = [feat for feat in high_corr_features if feat not in ['age', 'weight']]

# Define X and y
X = data_scaled[high_corr_features]
y = data_scaled['bmi']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best estimator
best_rf = grid_search.best_estimator_

# Functions for Image Processing
def get_facial_landmarks(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this path is correct
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    if len(rects) == 0:
        raise ValueError("No face detected in the image")
    
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    return shape, image

def compute_facial_metrics(landmarks):
    left_cheek = landmarks[1]
    right_cheek = landmarks[15]
    cheek_width = np.linalg.norm(left_cheek - right_cheek)
    
    left_jaw = landmarks[3]
    right_jaw = landmarks[13]
    jaw_width = np.linalg.norm(left_jaw - right_jaw)
    
    cheek_to_jaw_ratio = cheek_width / jaw_width if jaw_width != 0 else 0
    
    chin = landmarks[8]
    forehead = landmarks[27]
    face_height = np.linalg.norm(chin - forehead)
    
    face_width = cheek_width
    width_to_height_ratio = face_width / face_height if face_height != 0 else 0
    
    face_area = np.pi * (face_width / 2) * (face_height / 2)
    parameter_area_ratio = face_area  # Adjust as needed
    
    return {
        'WtoHRatio': width_to_height_ratio,
        'CheektoJawWidth': cheek_to_jaw_ratio,
        'ParameterAreaRatio': parameter_area_ratio
    }

def prepare_feature_vector(facial_metrics):
    # Initialize feature vector with mean values for all features used in training
    feature_vector = pd.DataFrame(columns=feature_cols)
    # Get mean values from the training data
    feature_means = data[feature_cols].mean()
    # Assign mean values
    feature_vector.loc[0] = feature_means
    # Fill in the facial metrics
    for feature in facial_metrics:
        if feature in feature_vector.columns:
            feature_vector.at[0, feature] = facial_metrics[feature]
    return feature_vector

def scale_features(feature_vector):
    # Scale the features using the previously fitted scaler
    scaled_features = scaler.transform(feature_vector)
    scaled_feature_vector = pd.DataFrame(scaled_features, columns=feature_vector.columns)
    return scaled_feature_vector

def predict_bmi(scaled_feature_vector):
    # Use only high_corr_features for prediction
    bmi_prediction_scaled = best_rf.predict(scaled_feature_vector[high_corr_features])
    # Inverse transform the BMI back to original scale
    bmi_original_scale = bmi_scaler.inverse_transform(bmi_prediction_scaled.reshape(-1, 1))
    return bmi_original_scale[0][0]

def estimate_bmi_from_image(image_path):
    landmarks, image = get_facial_landmarks(image_path)
    facial_metrics = compute_facial_metrics(landmarks)
    feature_vector = prepare_feature_vector(facial_metrics)
    scaled_feature_vector = scale_features(feature_vector)
    bmi_prediction = predict_bmi(scaled_feature_vector)
    print(f"Predicted BMI: {bmi_prediction:.2f}")
    return bmi_prediction

# Test with an image
predicted_bmi = estimate_bmi_from_image('test_face.jpg')
