import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Original dataset
data = {
    'Age': [30, 15, 65, None, 23],
    'Salary': [35500, 32000, None, 45000, 78000],
    'Department': ['HR', 'IT', 'IT', 'HR', 'HR'],
    'Purchased': ['Yes', 'No', 'No', 'Yes', 'Yes'],
    'Gender': ['Female', 'Female', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)
print("Before preprocessing dataset:\n", df)

# Handle missing values (mean imputation for Age & Salary)
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# Label encoding for all categorical columns
label_encoder = LabelEncoder()
for col in ['Department', 'Purchased', 'Gender']:
    df[col] = label_encoder.fit_transform(df[col])

# Standard scaling for numerical columns
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\nAfter preprocessing (only numerical data):\n", df)

# Splitting features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nX_train:\n", X_train)
print("\ny_train:\n", y_train)
