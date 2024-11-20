import matplotlib.pyplot as plt

# Function to plot loss and accuracy
def plot_metrics(train_losses, train_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.legend()
    plt.show()
