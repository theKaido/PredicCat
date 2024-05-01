import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

# Chargement des données
df = pd.read_csv('corpus.csv')

# Handling missing values
df.fillna(' ', inplace=True)  # Remplacez NaN par un espace

# Tokenisation des données
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_data = tokenizer(df['Description'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Encodage des genres de jeux
labels = pd.factorize(df['Genres'])[0]

# Création des datasets
dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], torch.tensor(labels))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Configuration du modèle
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))
optimizer = AdamW(model.parameters(), lr=2e-5,no_deprecation_warning=True)

# Entraînement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(5):  # nombre d'époques
    model.train()
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch {epoch+1}, loss : {loss.item()}")

# Sauvegarde du modèle
save_path = './ModeleTrainning'
if not os.path.exists(save_path):
    os.makedirs(save_path)
model.save_pretrained(save_path)
