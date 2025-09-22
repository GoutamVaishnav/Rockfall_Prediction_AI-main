import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data/raw/synthetic_data.csv')

X = df.drop(['Highwall', 'Risk_Level', 'Risk_Score'] ,axis=1)
y = df['Risk_Level']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Original Labels:", label_encoder.classes_)
print("Encoded Labels:", np.unique(y_encoded))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print("\nShape of training features:", X_train.shape)
print("Shape of testing features:", X_test.shape)

processed_data_dir = '../data/processed'
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

X_train.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
pd.Series(y_train).to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False, header=['rockfall_risk'])
pd.Series(y_test).to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False, header=['rockfall_risk'])

print("\nProcessed data saved successfully.")