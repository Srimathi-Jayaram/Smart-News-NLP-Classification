
import pandas as pd
import torch
import os
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", device)

# ✅ Load dataset
DATASET_PATH = "dataset/News_Category_Dataset_v2.csv"
assert os.path.exists(DATASET_PATH), f"❌ Dataset not found at {DATASET_PATH}"
df = pd.read_csv(DATASET_PATH)

# ✅ Prepare text and label
df["text"] = df["headline"] + " " + df["short_description"]
label_map = {label: idx for idx, label in enumerate(df["category"].unique())}
df["label"] = df["category"].map(label_map)

# ✅ Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ✅ Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ✅ Reduce size for quick training (remove these 4 lines for full training)
train_texts = train_texts[:500].reset_index(drop=True)
train_labels = train_labels[:500].reset_index(drop=True)
val_texts = val_texts[:200].reset_index(drop=True)
val_labels = val_labels[:200].reset_index(drop=True)

# ✅ Tokenizer and datasets
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_len=128)

# ✅ Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_map)
)
model.to(device)

# ✅ Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    max_steps=100,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    report_to="wandb",         # ✅ Logs training to wandb
run_name="bert-train-run"  # ✅ Names this run in wandb dashboard

)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ✅ Train
print("🚀 Starting training...")
trainer.train()

# ✅ Save everything
OUTPUT_DIR = "./bert_model_9labels"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f)

print(f"✅ Model, tokenizer, and label map saved to {OUTPUT_DIR}")

# Save id2label.json (reverse of label_map)
id2label = {v: k for k, v in label_map.items()}
with open("bert_model_9labels/id2label.json", "w") as f:
    json.dump(id2label, f)
    

