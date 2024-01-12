import torch.nn as nn
from torch.optim import Adam
import torch
from tqdm import tqdm 

def run_trainer(model, train_dataloader, test_dataloader, num_epochs, device, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_history = []

    model = model.to(device)

    for epoch in range(num_epochs):
        for i, (input_tensor, target_tensor) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            # Forward pass
            output = model(input_tensor)

            # Compute the loss
            loss = criterion(output.view(-1, output.shape[-1]), target_tensor.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
            # store train epoch loss
                loss_history.append([epoch, loss.item()])
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
    # evaluate on test loss on test dataset
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(test_dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            output = model(input_tensor, target_tensor)
            loss = criterion(output.view(-1, output.shape[-1]), target_tensor.view(-1))
            test_loss += loss.item()
    test_loss /= len(test_dataloader)
    print(f'Test Loss: {test_loss}')
    return loss_history


def generate_text(model, start_string, char_to_index, index_to_char, device, gen_length=100, temperature=0.1, ):
    model.eval()
    with torch.no_grad():
        input_text = start_string
        for _ in range(gen_length):
            input_tensor = torch.tensor([char_to_index[c] for c in input_text], dtype=torch.long).unsqueeze(0).to(device)
            output = model(input_tensor)
            output = output / temperature
            probs = torch.softmax(output[:, -1, :], dim=-1)
            next_char_index = torch.multinomial(probs, num_samples=1).squeeze().item()
            input_text += index_to_char[next_char_index]
    print(input_text)