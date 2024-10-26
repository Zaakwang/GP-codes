### Import necessary libraries and load the BERT model
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

### Define function for preprocessing
def replace_abbreviations(text):
    """Reveice a sentence, and replace the specific expressions to another forms."""
    abbreviations = {
        r'U\.S\. dollar': 'US dollar',
        r'U\.S': 'US',
        r'U\.K\.': 'UK'
    }
    for abbr, replacement in abbreviations.items():
        text = re.sub(abbr, replacement, text)
    return text

def clean_text(text):
    """Clean the sentence with meaningless symbols."""
    cleaned_text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~—%$]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,!?\'\s]', '', cleaned_text)
    text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

from nltk.tokenize import PunktSentenceTokenizer
def tokenizing(final):
    """Tokenize sentence."""
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(final)
    return sentences

def preprocessing(Ticker):
    """Combine the former functions, receive an Excel file that contains three columns, preprocess the text \
        and return a list of list that every sublist contains all processced text of a year."""
    df = pd.read_excel(f'{Ticker}.xlsx')
    item1 = df['Item 1'].values.tolist()
    item1a = df['Item 1A'].values.tolist()
    item7 = df['Item 7'].values.tolist()
    text = []
    all_text = []
    for i in range(20):
        text.append(str(item1[i]))
        text.append(str(item1a[i]))
        text.append(str(item7[i]))
        result_str = ' '.join(text)
        result_str = replace_abbreviations(result_str)
        result_str = clean_text(result_str)
        final = tokenizing(result_str)
        all_text.append(final)
        text = []
    return all_text

### Get embedding of the baseline sentence
definition = "Using digital technologies such as digital infrastructure, enterprise information systems, AI, automation, cybersecurity, IoT, cloud computing, big data, data analytics into their operations."
tokens2 = tokenizer.tokenize(definition)
input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)
with torch.no_grad():
        outputs2 = model(input_ids2)
        embeddings2 = outputs2[0][:, 0, :]

### Define a function to calculate sentence similarity for every sublist
def generate_dtcs(Ticker, batch_size=10):
    """Receive the Ticker of one company, preprocess the 10-K filings text of that company and return a DataFrame containing cosine similiarity \ 
        between every sentence and baseline sentence, each column represents a year."""

    cosinesim_year = []
    corpus = preprocessing(Ticker)

    for i in corpus:
        cosinesim = []
        # Process in batches
        for batch_start in range(0, len(i), batch_size):
            batch = i[batch_start:batch_start + batch_size]
            batch_cosinesim = []

            for j in batch:
                tokens1 = tokenizer.tokenize(j)
                # Limit token length to 128 in accordance to the requirement of BERT
                if len(tokens1) > 128:
                    tokens1 = tokens1[:128]
                input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)

                # Not optimizing the parameters of the model
                with torch.no_grad():
                    outputs1 = model(input_ids1)
                    embeddings1 = outputs1[0][:, 0, :]

                # Compute the cosine similarity
                similarity_score = cosine_similarity(embeddings1, embeddings2)
                batch_cosinesim.append(similarity_score)

                # Clear memory after each sentence
                del input_ids1, embeddings1, outputs1
                torch.cuda.empty_cache()

            cosinesim.extend(batch_cosinesim)

        cosinesim = [arr.tolist()[0][0] for arr in cosinesim]
        cosinesim_year.append(cosinesim)

    # Calculate max_len only for the available years
    max_len = max(len(year) for year in cosinesim_year)

    # Create DataFrame and fill with zeros where necessary
    df = pd.DataFrame({
        str(2023 - idx): pd.Series(cosinesim_year[idx]).reindex(range(max_len), fill_value=0)
        for idx in range(len(cosinesim_year))
    })

    return df