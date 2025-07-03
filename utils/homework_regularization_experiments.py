import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 20
learning_rate = 0.001

# Загрузка данных CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Базовая архитектура сети
class BaseNet(nn.Module):
    def __init__(self, dropout_rate=0.0, use_batchnorm=False, weight_decay=0.0):
        super(BaseNet, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.weight_decay = weight_decay

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        if use_batchnorm:
            self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)

        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        return x


# Функция для обучения и оценки модели
def train_and_evaluate(model, train_loader, test_loader, num_epochs, learning_rate, weight_decay=0.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Оценка на тестовом наборе
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return train_losses, test_accuracies


# Функция для визуализации весов
def plot_weight_distributions(models, model_names):
    num_models = len(models)
    rows = (num_models + 2) // 3  # Вычисляем необходимое количество строк

    plt.figure(figsize=(15, 5 * rows))
    for i, (model, name) in enumerate(zip(models, model_names)):
        weights = []
        for param in model.parameters():
            if param.dim() > 1:  # Игнорируем bias
                weights.extend(param.view(-1).cpu().detach().numpy())

        plt.subplot(rows, 3, i + 1)
        plt.hist(weights, bins=50, alpha=0.7)
        plt.title(f"Weight Distribution: {name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# 3.1 Сравнение техник регуляризации
def compare_regularization_techniques():
    techniques = [
        ("No regularization", {'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("Dropout 0.1", {'dropout_rate': 0.1, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("Dropout 0.3", {'dropout_rate': 0.3, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("Dropout 0.5", {'dropout_rate': 0.5, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("BatchNorm only", {'dropout_rate': 0.0, 'use_batchnorm': True, 'weight_decay': 0.0}),
        ("Dropout 0.3 + BatchNorm", {'dropout_rate': 0.3, 'use_batchnorm': True, 'weight_decay': 0.0}),
        ("L2 regularization", {'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.001}),
    ]

    results = {}
    models = []

    for name, params in techniques:
        print(f"\nTraining {name}")
        model = BaseNet(**params).to(device)
        train_losses, test_accuracies = train_and_evaluate(
            model, train_loader, test_loader, num_epochs, learning_rate, params['weight_decay']
        )
        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1]
        }
        models.append(model)

    # Визуализация результатов
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['train_losses'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, res in results.items():
        plt.plot(res['test_accuracies'], label=name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Визуализация распределения весов
    plot_weight_distributions(models, [name for name, _ in techniques])

    # Вывод финальных точностей
    print("\nFinal Accuracies:")
    for name, res in results.items():
        print(f"{name}: {res['final_accuracy']:.2f}%")


# 3.2 Адаптивная регуляризация
class AdaptiveNet(nn.Module):
    def __init__(self, initial_dropout=0.1, final_dropout=0.5,
                 batchnorm_momentum=0.1, layer_specific_reg=False):
        super(AdaptiveNet, self).__init__()
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.batchnorm_momentum = batchnorm_momentum
        self.layer_specific_reg = layer_specific_reg

        # Конволюционные слои
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=batchnorm_momentum)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=batchnorm_momentum)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.bn3 = nn.BatchNorm1d(512, momentum=batchnorm_momentum)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(512, 10)

        # Инициализация dropout
        self.current_dropout_rate = initial_dropout
        self.dropout1 = nn.Dropout2d(initial_dropout)
        self.dropout2 = nn.Dropout2d(initial_dropout)
        self.dropout3 = nn.Dropout(initial_dropout)

    def update_dropout(self, epoch, total_epochs):
        # Линейное увеличение dropout rate
        self.current_dropout_rate = self.initial_dropout + (self.final_dropout - self.initial_dropout) * (
                epoch / total_epochs)
        self.dropout1.p = self.current_dropout_rate
        self.dropout2.p = self.current_dropout_rate
        self.dropout3.p = self.current_dropout_rate

        if self.layer_specific_reg:
            # Разные dropout rates для разных слоев
            self.dropout1.p = min(0.2, self.current_dropout_rate)
            self.dropout2.p = min(0.4, self.current_dropout_rate)
            self.dropout3.p = min(0.3, self.current_dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        return x


def evaluate_adaptive_techniques():
    techniques = [
        ("Fixed Dropout 0.3", {'initial_dropout': 0.3, 'final_dropout': 0.3,
                               'batchnorm_momentum': 0.1, 'layer_specific_reg': False}),
        ("Increasing Dropout 0.1-0.5", {'initial_dropout': 0.1, 'final_dropout': 0.5,
                                        'batchnorm_momentum': 0.1, 'layer_specific_reg': False}),
        ("High BatchNorm Momentum 0.9", {'initial_dropout': 0.3, 'final_dropout': 0.3,
                                         'batchnorm_momentum': 0.9, 'layer_specific_reg': False}),
        ("Layer-specific Dropout", {'initial_dropout': 0.1, 'final_dropout': 0.5,
                                    'batchnorm_momentum': 0.1, 'layer_specific_reg': True}),
        ("Combined Adaptive", {'initial_dropout': 0.1, 'final_dropout': 0.4,
                               'batchnorm_momentum': 0.5, 'layer_specific_reg': True}),
    ]

    results = {}

    for name, params in techniques:
        print(f"\nTraining {name}")
        model = AdaptiveNet(**params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            model.update_dropout(epoch, num_epochs)

            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            # Оценка на тестовом наборе
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(
                f"Epoch {epoch + 1}, Dropout: {model.current_dropout_rate:.2f}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1]
        }

    # Визуализация результатов
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['train_losses'], label=name)
    plt.title('Training Loss (Adaptive Techniques)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, res in results.items():
        plt.plot(res['test_accuracies'], label=name)
    plt.title('Test Accuracy (Adaptive Techniques)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Вывод финальных точностей
    print("\nFinal Accuracies (Adaptive Techniques):")
    for name, res in results.items():
        print(f"{name}: {res['final_accuracy']:.2f}%")


# Запуск экспериментов
if __name__ == "__main__":
    print("=== 3.1 Сравнение техник регуляризации ===")
    compare_regularization_techniques()

    print("\n=== 3.2 Адаптивная регуляризация ===")
    evaluate_adaptive_techniques()