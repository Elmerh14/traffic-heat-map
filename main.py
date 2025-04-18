# train_crash_classifier.py (Continued Training from v2 → Save as v3)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import os

# === CONFIG ===
USE_CSV = True  # Set to False to use the full Excel file
FILE_PATH = "heatmap_input.csv" if USE_CSV else "CleanData.xlsx"
TEXT_COL = "input_text"
LABEL_COL = "Crash Cause"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 2


# === 1. Load & Clean Data ===
read_kwargs = {"nrows": 50} if not USE_CSV else {}
df = pd.read_csv(FILE_PATH) if USE_CSV else pd.read_excel(FILE_PATH, **read_kwargs)

def get_next_model_version(base_name="crash_cause_model"):
    version = 2
    while os.path.exists(f"{base_name}_v{version}"):
        version += 1
    model_path = f"{base_name}_v{version}"
    return model_path, version

SAVE_MODEL_PATH, version = get_next_model_version()
PREVIOUS_MODEL_PATH = f"crash_cause_model_v{version - 1}"

def get_primary_cause(row):
    for col in ['TU-1 Human Contributing Factor', 'TU-2 Human Contributing Factor']:
        val = row[col]
        if pd.notna(val) and val not in ["No Apparent Contributing Factor", "Not Observed"]:
            return val
    return None

if LABEL_COL not in df.columns:
    df[LABEL_COL] = df.apply(get_primary_cause, axis=1)
    df = df[df[LABEL_COL].notna()]

if TEXT_COL not in df.columns:
    def create_sentence(row):
        return (
            f"In {row['City'] if pd.notna(row['City']) else 'Unknown City'}, "
            f"{row['County']} County on {row['Crash Date']} at {row['Crash Time']}, "
            f"a {row['Crash Type']} crash occurred. "
            f"Road was {row['Road Condition']}, weather was {row['Weather Condition']}, "
            f"lighting was {row['Lighting Conditions']}. "
            f"Driver actions: {row['TU-1 Driver Action']}, {row['TU-2 Driver Action']}. "
            f"Driver ages: {row['TU-1 Age']}, {row['TU-2 Age']}."
        )
    df[TEXT_COL] = df.apply(create_sentence, axis=1)

# === 2. Encode Labels (Load existing encoder) ===
import joblib
le = joblib.load("label_encoder_v2.pkl")
print(f"✅ Loaded label encoder with {len(le.classes_)} classes: {list(le.classes_)}")

# Filter to only known labels
df = df[df[LABEL_COL].isin(le.classes_)]
df['label'] = le.transform(df[LABEL_COL])
num_labels = len(le.classes_)

# === 3. Tokenize ===
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class CrashDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# === 4. Dataset Split ===
dataset = CrashDataset(df[TEXT_COL], df['label'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# === 5. Load Model (from v2) ===
model = DistilBertForSequenceClassification.from_pretrained(PREVIOUS_MODEL_PATH, num_labels=num_labels)

# === 6. Training ===
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# === 7. Save Model ===
model.save_pretrained(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)

print(f"✅ Continued training complete. Model saved to {SAVE_MODEL_PATH}")
