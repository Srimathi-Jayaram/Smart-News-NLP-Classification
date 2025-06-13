
import pandas as pd
import torch
import os
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load dataset (Update path if needed)
DATASET_PATH = "dataset/News_Category_Dataset_v2.csv"
assert os.path.exists(DATASET_PATH), f"Dataset not found at {DATASET_PATH}"
df = pd.read_csv(DATASET_PATH)

# ✅ Create full text column
df["text"] = df["headline"] + " " + df["short_description"]
print("Sample categories:", df['category'].value_counts().head())

# ✅ Encode labels
label_map = {label: idx for idx, label in enumerate(df['category'].unique())}
df['label'] = df['category'].map(label_map)

# ✅ Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ✅ Stratified Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# ✅ Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Optional: Swap to RoBERTa for better performance
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# ✅ Dataset objects
train_dataset = NewsDataset(train_texts.reset_index(drop=True), train_labels.reset_index(drop=True), tokenizer, max_len=128)
val_dataset = NewsDataset(val_texts.reset_index(drop=True), val_labels.reset_index(drop=True), tokenizer, max_len=128)

# ✅ Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
# model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_map))
model.to(device)

# ✅ Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ✅ Training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_total_limit=2,
    gradient_accumulation_steps=1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ✅ Train
print("🚀 Starting training...")
trainer.train()

# ✅ Save model + tokenizer
OUTPUT_DIR = "./bert_model_9labels"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ✅ Save label map for inference
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f)

print("✅ Model and tokenizer saved to", OUTPUT_DIR)



# ✅ Optional: Softmax prediction function (use this in inference script)
def get_predictions(text_list):
    model.eval()
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        categories = [list(label_map.keys())[list(label_map.values()).index(pred.item())] for pred in predictions]
        return list(zip(text_list, categories, confidences.cpu().numpy()))
