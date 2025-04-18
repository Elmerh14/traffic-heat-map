import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample Data Placeholder (you'll replace this with your friend's processed file)
data = pd.DataFrame({
    'description': [
        "Driver was speeding in icy conditions.",
        "Collision due to distracted driving while texting.",
        "Hit a deer at night, road was poorly lit.",
        "Driver was under the influence of alcohol.",
        "Foggy weather caused reduced visibility and crash."
    ],
    'label': [1, 2, 3, 4, 5]  # Sample numeric labels
})

# Tokenizer and Encoding
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class CrashDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()} | {'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['description'], data['label'], test_size=0.2, random_state=42
)

train_dataset = CrashDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = CrashDataset(val_texts.tolist(), val_labels.tolist())

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)

# Training Loop (simplified)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
