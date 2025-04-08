import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data/Mental_Health_Lifestyle_Dataset.csv")

# Separate target and features
target = 'Happiness Score'
X = data.drop(columns=[target])
y = data[target]

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define categorical and numeric columns. Adjust these lists as needed.
categorical_features = [
    'Country', 'Gender', 'Exercise Level', 'Diet Type', 'Stress Level', 'Mental Health Condition'
]
numeric_features = [col for col in X.columns if col not in categorical_features]

# Create a pipeline for numeric features:
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create a pipeline for categorical features:
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # in case of missing values
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine numeric and categorical transformations into a preprocessor:
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create the final pipeline:
clf = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit the pipeline on training data and transform both train and test sets:
X_train_transformed = clf.fit_transform(X_train)
X_test_transformed = clf.transform(X_test)

# Convert the transformed output to DataFrame.
# If the pipeline returns a sparse matrix, convert it to dense first.
def to_dataframe(X, original_index):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, index=original_index)

train_transformed_df = to_dataframe(X_train_transformed, X_train.index)
test_transformed_df = to_dataframe(X_test_transformed, X_test.index)

# Append the target variable to the corresponding DataFrames:
train_transformed_df['y'] = y_train.values
test_transformed_df['y'] = y_test.values

# Save the processed data as CSV files:
train_transformed_df.to_csv('data/processed_train_data.csv', index=False)
test_transformed_df.to_csv('data/processed_test_data.csv', index=False)

# Save the entire pipeline to a pickle file:
with open('data/pipeline.pkl', 'wb') as f:
    pickle.dump(clf, f)
