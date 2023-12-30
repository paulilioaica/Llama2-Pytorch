from datasets import load_dataset
from tokenizer import tokenize_function
from dataset import TextDataset
from torch.utils.data import DataLoader

def run():

    dataset_train = load_dataset('ag_news', split="train[10%:20%]")
    dataset_test = load_dataset('ag_news', split="test[10%:30%]") 

    train_dataset = dataset_train
    test_dataset = dataset_test


    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


    train_dataset = TextDataset(tokenized_train_dataset)
    test_dataset = TextDataset(tokenized_test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)