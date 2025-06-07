import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from scripts.cnn_model import EmotionCNN
from scripts.crema_dataset import CREMADataset

def train_model():
    dataset = CREMADataset("data/crema_labels.csv")

    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    model = EmotionCNN(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "models/emotion_cnn.pth")
    print(" Model saved to models/emotion_cnn.pth")

if __name__ == "__main__":
    train_model()
