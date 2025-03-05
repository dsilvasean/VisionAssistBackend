import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

project_root = "/home/sean/repos/project_repos/VisionAssistBackend"
dataset_path = os.path.join(project_root, 'data/dataset1.csv')
label_encoder_path = os.path.join(project_root, 'data/label_encoder.pkl')

# Load dataset and fit LabelEncoder
df = pd.read_csv(dataset_path)
label_encoder = LabelEncoder()
label_encoder.fit(df['intent'])

# Save the entire LabelEncoder object
pd.to_pickle(label_encoder, label_encoder_path)
print("LabelEncoder saved with classes:", label_encoder.classes_)
