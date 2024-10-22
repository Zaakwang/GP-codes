# import necessary libraries
import requests
import pandas as pd

# create request header
headers = {}
companyTickers = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=headers
    )

# changing cik format for requesting
companyData = pd.DataFrame.from_dict(companyTickers.json(), orient='index')
companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10)

# Search CIK based on ticker list
def search_cik(ticker):
    selected_df = companyData[companyData['ticker']==ticker]
    cik = selected_df.cik_str[0]
    return cik

# Get company facts data
def get_data(cik):
    companyFacts = requests.get(
    f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
    headers=headers
    )
    available_items = companyFacts.json()['facts']['us-gaap'].keys()
    available_items = list(available_items)
    return available_items

# Reverse list
def reverse_list(lst):
    return lst[::-1]

# Padding list
def pad_list(lst, length=20, pad_value=0):
    num_to_add = length - len(lst)
    if num_to_add > 0:
        lst.extend([pad_value] * num_to_add)
    return lst

# Get targeted list
def get_items_data(cik: str, items: str) -> list: 
    companyConcept = requests.get(
    (
    f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}'
     f'/us-gaap/{items}.json'
    ),
    headers=headers
    )
    Data = pd.DataFrame.from_dict((
               companyConcept.json()['units']['USD']))
    T10K = Data[Data.form == '10-K']
    T10K = T10K.reset_index(drop=True)
    if len(T10K) > 20:
        target = T10K['val'][-20:].values.tolist()
        target = reverse_list(target)
    else:
        target = T10K['val'].values.tolist()
        target = reverse_list(target)
        target = pad_list(target)
    return target