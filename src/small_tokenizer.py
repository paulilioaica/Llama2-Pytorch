from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
