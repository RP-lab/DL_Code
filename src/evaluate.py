import torch
from model import CNNModel
from data_preparation import prepare_data


def evaluate_model(data_dir, model_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = prepare_data(data_dir, batch_size)

    model = CNNModel(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    evaluate_model(data_dir="../data/test", model_path="../models/defect_detection_cnn.pth")
