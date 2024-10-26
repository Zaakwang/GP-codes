### Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR

from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup

from tqdm import tqdm, trange

import warnings
warnings.filterwarnings("ignore")

### Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

### Load training dataset
df = pd.read_csv()
df = df[['sentence', 'Label']]

### Add [CLS] and [SEP] tokens to sentences, split training and test set
sentences = df.sentence.values

sentences = ["[CLS] " + sen + " [SEP]" for sen in sentences]
labels = df.Label.values

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size = 0.2, random_state=666)

### Tokenize sentences and transform tokens into IDs
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
sentence_tokens = [tokenizer.tokenize(sen) for sen in X_train]
max_len = 128
sentence_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentence_tokens]
sentence_ids = pad_sequences(sentence_ids, maxlen=max_len, dtype='long', truncating='post', padding='post') # Limit token lengths to 128

### Genterate attention mask
attention_mask = [[1 if id> 0 else 0 for id in sen] for sen in sentence_ids]

### Split training and eval set
X_train_1, X_eval_1, y_train_1, y_eval_1 = train_test_split(sentence_ids, y_train, test_size=0.2, random_state=666)
train_masks, eval_masks, _, _ = train_test_split(attention_mask, sentence_ids, test_size=0.2, random_state=666)

### Transform data to tensor
X_train = torch.tensor(X_train_1)
X_eval = torch.tensor(X_eval_1)
y_train = torch.tensor(y_train_1)
y_eval = torch.tensor(y_eval_1)
train_masks = torch.tensor(train_masks)
eval_masks = torch.tensor(eval_masks)

### Pack data for batch training
batch_size = 32

train_dataset = TensorDataset(X_train, train_masks, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

eval_dataset = TensorDataset(X_eval, eval_masks, y_eval)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

### Training
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# Epochs
EPOCHS = 5
# Optimizers
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
# Learning rate
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader)*EPOCHS)

# Define function to calculate accuracy
def accuracy(labels, preds):
    """Receive labels and preds, return accuracy rate."""
    preds = np.argmax(preds, axis=1).flatten() # shape = (1, :)
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(preds) # Accuracy rate
    return acc

# main
if __name__ == '__main__':
    train_loss = []


    for i in tqdm(range(EPOCHS), desc='Epoch'):

        model.train()

        tr_loss = 0
        tr_examples = 0
        tr_steps = 0

        for i, batch_data in enumerate(train_dataloader):
            # Load data batch by batch
            batch_data = tuple(data.to(device) for data in batch_data) # Deploy to GPU
            # Parsing
            inputs_ids, inputs_masks, inputs_labels = batch_data
            # Set gradient to zero
            optimizer.zero_grad()
            # Feedforward propagation
            outputs = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks, labels=inputs_labels)
            # Get loss
            loss = outputs['loss']
            # Save loss
            train_loss.append(loss.item())
            # Accmulate loss
            tr_loss += loss.item()
            # Acculate samples
            tr_examples += inputs_ids.size(0)
            # Batch nubmers
            tr_steps += 1
            # Backward propagationi
            loss.backward()
            # Update parameters
            optimizer.step()
            # Update learning rates
            scheduler.step()


        print("Training loss : {}".format(tr_loss / tr_steps))


        # Set model to eval mode
        model.eval()
        eval_acc = 0.0, 0.0
        eval_steps, eval_examples = 0.0, 0.0

        for batch in eval_dataloader:
            # Deploy to GPU
            batch = tuple(data.to(device) for data in batch_data)
            # Parsing
            inputs_ids, inputs_masks, inputs_labels = batch
            # No upgrading gradients
            with torch.no_grad():
                preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)
            # Deploy to CPU to calculate accuracy
            preds = preds['logits'].detach().to('cpu').numpy() 
            labels = inputs_labels.to('cpu').numpy()
            # Compute accuracy
            eval_acc += accuracy(labels, preds)

            eval_steps += 1

        print("Eval Accuracy : {}".format(eval_acc / eval_steps))

        print("\n\n")

### Visualize training loss
plt.figure(figsize=(12,10))
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss)
plt.show()

### Tokenize test set sentences
sentences_tokens = [tokenizer.tokenize(sen) for sen in X_test]
sentence_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentences_tokens] # tokens to ids
sentence_ids = pad_sequences(sentence_ids, maxlen=max_len, dtype='long', truncating='post', padding='post') # limit token lengths
attention_mask = [[1 if id > 0 else 0 for id in sen] for sen in sentence_ids] # Generate attention mask

### Transform data to tensor
sentence_ids = torch.tensor(sentence_ids)
attention_mask = torch.tensor(attention_mask)
labels = torch.tensor(y_test)

### Pack data and load data
test_dataset = TensorDataset(sentence_ids, attention_mask, labels)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

### Test model performance
from sklearn.metrics import confusion_matrix
import numpy as np

model.eval()

test_loss, test_acc = 0.0, 0.0
steps = 0
num = 0

all_preds = []
all_labels = []

for batch in test_dataloader:
    batch = tuple(data.to(device) for data in batch)
    inputs_ids, inputs_masks, inputs_labels = batch
    with torch.no_grad():
        preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)
    preds = preds['logits'].detach().to('cpu').numpy()
    inputs_labels = inputs_labels.to('cpu').numpy()

    all_preds.append(np.argmax(preds, axis=1))
    all_labels.append(inputs_labels)

    acc = accuracy(inputs_labels, preds)
    test_acc += acc
    steps += 1
    num += len(inputs_ids)

print("steps = ", steps)
print("test number = ", num)
print("test acc : {}".format(test_acc / steps))

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", conf_matrix)

### Define function to transform logits result from predictions to actual labels
def classify_logits(logits_list):
    """Receive the logits result from predictions and return a list that contains the actual labels from a batch."""
    results = []

    # Iterate over each pair of logits in the list
    for logits in logits_list:
        # Apply softmax to get probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits))

        # Determine the predicted class (0 or 1)
        predicted_class = np.argmax(probabilities)

        # Append the result (0 or 1) to results list
        results.append(predicted_class)

    return results

### Predicts sentences from company 10-K filings
company_list = []
for com in company_list:
  df = pd.read_excel(f'{com}pe.xlsx')
  list_of_list = [df[col].tolist() for col in df.columns]
  list_of_list = list_of_list[1:]
  total_list = []
  for i in list_of_list:
    year_list = []
    sentences = ["[CLS] " + sen + " [SEP]" for sen in i]
    sentences_tokens = [tokenizer.tokenize(sen) for sen in i]
    sentence_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentences_tokens]
    sentence_ids = pad_sequences(sentence_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')
    attention_mask = [[1 if id > 0 else 0 for id in sen] for sen in sentence_ids]
    sentence_ids = torch.tensor(sentence_ids)
    attention_mask = torch.tensor(attention_mask)
    test_dataset = TensorDataset(sentence_ids, attention_mask)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    for batch in test_dataloader:
      # 部署到 GPU
      batch = tuple(data.to(device) for data in batch)
      # 数据解包
      inputs_ids, inputs_masks = batch
      # 验证阶段不需要计算梯度
      with torch.no_grad():
          preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)
      preds = preds['logits'].detach().to('cpu').numpy()
      preds = preds.tolist()
      preds = classify_logits(preds)
      for j in preds:
        year_list.append(j)
    total_list.append(year_list)
  df1 = pd.DataFrame.from_records(zip(*total_list))
  df1.to_excel(f'{com}class.xlsx')