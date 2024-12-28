import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Load the Excel file
file_path = '/home/data3/Ali/Code/Moghis/Test-data-2.xlsx'
data = pd.read_excel(file_path)

# Exclude specific columns ('uid' and 'composite_score') from processing
columns_to_exclude = ['uid', 'composite_score']

# Separate categorical and numerical features, excluding specified columns
categorical_features = data.select_dtypes(include=['object']).columns.difference(columns_to_exclude)
numerical_features = data.select_dtypes(exclude=['object']).columns.difference(columns_to_exclude)

# Drop rows with all NaN values in numerical features
data = data.dropna(how='all', subset=numerical_features)

# Identify and drop completely NaN columns from numerical features
valid_numerical_features = numerical_features[data[numerical_features].notna().any()]

# Handle missing values with KNN imputation for valid numerical features
numerical_data = data[valid_numerical_features].copy()
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(numerical_data)

# Reassign the imputed data back to the DataFrame with proper index alignment
data[valid_numerical_features] = pd.DataFrame(imputed_data, 
                                              columns=valid_numerical_features, 
                                              index=data.index)

# Encode categorical features
encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = encoder.fit_transform(data[feature].astype(str))

# Reset index to ensure consistent row count
data.reset_index(drop=True, inplace=True)

# Save the preprocessed data to a new file
output_file = '/home/data3/Ali/Code/Moghis/Test-data-preprocess-2.xlsx'
data.to_excel(output_file, index=False)

print(f"Preprocessing complete. Preprocessed data saved to {output_file}")
