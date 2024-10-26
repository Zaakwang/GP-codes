### Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

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

### Define a dictionary
digital_transformation_dict = {
    'analytics': r'\banalytics\b|\banalysis\b',
    'virtual reality': r'\baugmented reality\b|\bvirtual reality\b', 
    'automation': r'\bautomation solutions\b|\bintelligent automation\b|\bprocess automation\b|\brobotic process automation\b|\bautonomous?[-]?tech\b|\bautomation\b',
    'artificial intelligence': r'\bartificial?[-]?intelligence\b|\bai?[-]?tech\b|\bai?[-]?related\b|\bconversational ai\b|\bevolutionary ai\b|\bevolutionary computing\b|\bintelligent?[-]?system\b|\bcomputer?[-]?vision\b|\bvirtual agent\b|\bvirtual? [-]?assistant\b|\bartificial intelligence\b|\bai\b', 
    'neural network': r'\bneural network\b|\bdeep learning\b|\bnatural language processing\b|\blarge language model\b|\bneural networks\b|\btransformers\b|\bllm\b|\bgpt\b',
    'machine learning': r'\bbiometric\b|\bmachine?[-]?learning\b|\bimage?[-]?recognition\b|\bfacial?[-]?recognition\b|\bspeech?[-]?recognition\b|\bcognitive computing\b|\bmachine learning\b', 
    'big data': r'\bbig?[-]?data\b|\bsmart?[-]?data\b|\bdata?[-]?science\b|\bdata?[-]?mining\b|\bdata lake\b|\bdevops\b|\bdigital twin\b|\bedge computing\b|\bsaas\b|\bbig data\b|\bdata science|\bdata mining|\bdata analysis\b', 
    'cloud': r'\bcloud?[-]?platform\b|\bcloud?[-]?based\b|\bcloud?[-]?computing\b|\bcloud?[-]?deployment\b|\bcloud enablement\b|\bhybrid cloud\b|\bvirtual?[-]?machine\b|\bapi\b|\bblockchain\b|\bblockchains\b',
    'digital strategy': r'\bdigital?[-]?transformation\b|\bdigital?[-]?revolution\b|\bdigital?[-]?strategy\b|\bdigital?[-]?marketing\b|\bbusiness intelligence\b|\bcustomer intelligence\b|\boperating intelligence\b|\bdigital vision\b|\bdigital culture\b|\bdata governance\b|\bIT value\b|\bbusiness IT relationship\b|\bdigital maturity\b',  
    'e-commerce': r'\be[-]?commerce\b|\be[-]?business\b|\be-commerce\b|\be-business\b',
    'IoT': r'\bIoT\b|\bInternet of Things\b',
    'database': r'\b(rdbms|relational database|nosql|sql database|non-relational database|database management system|dbms|sql server|oracle database | mysql|postgresql|mongodb|cassandra|dynamodb|redshift|data warehouse|data warehousing|olap|oltp|query optimization|database security|database backup|acid compliance|sharding|database scaling|horizontal scaling|vertical scaling)\b'
}

### Define a function to count the term frequency
def count_topic_occurrences(sentences_list, dt_dict):
    """Receive the list of list and dictionary that is defined above, count the term frequency for each key in the dictionary, and return a DataFrame \
        that contains the count of term frequency for every key in the dictionary."""

    all_topic_counts = []
    
    for idx, sentences in enumerate(sentences_list):
        topic_counts = {topic: 0 for topic in dt_dict}

        for sentence in sentences:
            for topic, pattern in dt_dict.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    topic_counts[topic] += 1

        topic_counts['Group'] = f'Group_{idx + 1}' 

    df = pd.DataFrame(all_topic_counts)
    return df