# crash_heatmap_generator.py

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import folium
from folium.plugins import HeatMap

# === Load trained model and tokenizer ===
model_path = "./crash_cause_model"
model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    model_path,
    local_files_only=True
)

model.eval()

# === Load label encoder ===
label_encoder = joblib.load("label_encoder.pkl")

# === Load hypothetical or real location-time-weather data ===
data = pd.read_csv("heatmap_input.csv")  # You create this CSV manually or simulate it

# Create input sentences like the ones used during training
def create_sentence(row):
    return (
        f"In {row['City']}, {row['County']} County on {row['Crash Date']} at {row['Crash Time']}, "
        f"a {row['Crash Type']} crash occurred. "
        f"Road was {row['Road Condition']}, weather was {row['Weather Condition']}, "
        f"lighting was {row['Lighting Conditions']}. "
        f"Driver actions: {row['TU-1 Driver Action']}, {row['TU-2 Driver Action']}. "
        f"Driver ages: {row['TU-1 Age']}, {row['TU-2 Age']}."
    )

# === Predict crash causes ===
data['input_text'] = data.apply(create_sentence, axis=1)
tokens = tokenizer(data['input_text'].tolist(), padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=1)
    predicted_labels = label_encoder.inverse_transform(predictions.numpy())

# Add predictions to data
data['Predicted Crash Cause'] = predicted_labels

# === Normalize crash risk for heatmap (you can modify this logic) ===
risk_weights = {
    'Aggressive Driving': 1.0,
    'Distracted Driving': 0.9,
    'Fatigue': 0.8,
    'Speeding': 0.9,
    'Inexperience': 0.6,
    'Illness': 0.5,
    'Unknown': 0.3
    # Add more based on your dataset's label classes
}

def get_risk_score(cause):
    return risk_weights.get(cause, 0.4)  # Default risk score

data['Risk Score'] = data['Predicted Crash Cause'].apply(get_risk_score)

# === Generate Heatmap ===
map_center = [39.7392, -104.9903]  # Default center (Denver, CO)
m = folium.Map(location=map_center, zoom_start=11)

heat_data = [
    [row['Latitude'], row['Longitude'], row['Risk Score']] 
    for _, row in data.iterrows() if not pd.isna(row['Latitude']) and not pd.isna(row['Longitude'])
]

HeatMap(heat_data, radius=10, blur=15).add_to(m)
m.save("crash_risk_heatmap.html")

print("✅ Heatmap saved as crash_risk_heatmap.html")
