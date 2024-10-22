### Import necessary libraries
pip install edgartools

### Pre-set condition
from edgar import Company, set_identity
import pandas as pd
set_identity()

### Create company list
company_list = []

### Get qualiative data
for i in company_list:
    company = Company(i)
    filings = company.get_filings(form = '10-K')
    
    df = pd.DataFrame({
        'Item 1': [],
        'Item 1A': [],
        'Item 7': []
    })

    for j in range(min(20, len(filings))):
        targeting_filings = filings[j]
        target_filings = targeting_filings.obj()  
    
        Item1 = target_filings['Item 1']
        Item1a = target_filings['Item 1A']
        Item7 = target_filings['Item 7']
    
        new_row = pd.DataFrame({
            'Item 1': [Item1], 
            'Item 1A': [Item1a], 
            'Item 7': [Item7]
        })
    
        df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_excel(f'{i}.xlsx', index = False)