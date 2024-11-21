import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from model import CNNModel
from data_preparation import prepare_data

def train_model(data_dir, save_path, num_epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the directory exists and has write permissions
    if not os.access(os.path.dirname(save_path), os.W_OK):
        print(f"Error: The directory {os.path.dirname(save_path)} is not writable!")
        return

    train_loader, val_loader, _, classes = prepare_data(data_dir, batch_size)
    model = CNNModel(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model(data_dir="../data/train", save_path="E:/Project/DL_gradedAssign/models/defect_detection_cnn.pth")
