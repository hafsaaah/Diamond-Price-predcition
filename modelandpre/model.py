import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

# Assume these are your numerical and categorical columns
numerical_cols = ['carat', 'depth', 'table']
categorical_cols = ['cut', 'color', 'clarity']

# Define the category order for OrdinalEncoder
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# Numerical Pipeline
num_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# Categorical Pipeline
cat_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
        ('scaler', StandardScaler())
    ]
)

# Combine both pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_pipeline', num_pipeline, numerical_cols),
        ('cat_pipeline', cat_pipeline, categorical_cols)
    ]
)
data = pd.read_csv('c:/Users/hafsa/OneDrive/Documents/miniprojecthafsa/modelandpre/training_datanew.csv')
# Split features and target
X = data[numerical_cols + categorical_cols]
y = data['price']  # Assuming 'price' is the target variable

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

# Preprocess the training and test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train the XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(X_train_transformed, y_train)

# Save the preprocessor to a pickle file
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Save the trained model to a pickle file
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
