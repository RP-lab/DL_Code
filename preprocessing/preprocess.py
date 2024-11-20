import spacy
from torchtext.data.utils import get_tokenizer

def preprocess_text(text):
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    tokenized = tokenizer(text)
    return tokenized
