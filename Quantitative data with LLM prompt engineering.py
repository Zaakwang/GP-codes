### Import OpenAI
from openai import OpenAI
client = OpenAI()

### Define a function to request prompt result
def check_topic(sentence):
    """Receive a text and return the binary classification resulto indicate if the text is related to the topic."""

    prompt = f"""
    You are an AI language model and you are doing research in the field of operational management. Your task is to classify the following sentences from the 10-K filings of listed companies. If the sentence's concept is closely related to , return 1. Otherwise, return 0.
    
    Sentence: "{sentence}"
    
    Return 1 for  related, or 0 for not related.
    """

    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    result = response.choices[0].message.content.strip()
    
    return int(result)
