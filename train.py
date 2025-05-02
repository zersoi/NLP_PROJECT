import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from wordcloud import WordCloud
import csv

rows = []
with open('/content/sentiment140.csv', encoding='latin-1') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 6:
            rows.append([row[0], row[5]])  

df = pd.DataFrame(rows, columns=['label', 'text'])
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df['label'] = df['label'].replace(4, 1)
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.strip() != '']

positive = df[df['label'] == 1]
negative = df[df['label'] == 0]
n_samples = min(len(positive), len(negative), 5000)
positive = positive.sample(n_samples)
negative = negative.sample(n_samples)
balanced_df = pd.concat([positive, negative]).sample(frac=1).reset_index(drop=True)

train_texts = balanced_df['text'].iloc[:int(0.8 * len(balanced_df))].tolist()
train_labels = balanced_df['label'].iloc[:int(0.8 * len(balanced_df))].tolist()
test_texts = balanced_df['text'].iloc[int(0.8 * len(balanced_df)):].tolist()
test_labels = balanced_df['label'].iloc[int(0.8 * len(balanced_df)):].tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()
for epoch in range(epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("Model and tokenizer saved to 'saved_model' folder.")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

positive_texts_list = [text for idx, text in enumerate(test_texts) if test_labels[idx] == 1]

if len(positive_texts_list) > 0:
    positive_texts = " ".join(positive_texts_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud for Positive Tweets')
    plt.show()
else:
    print("No positive texts found to generate a WordCloud.")
