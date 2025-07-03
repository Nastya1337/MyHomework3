import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import json
import os

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Функция для создания модели из конфигурации
def create_model_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    layers = []
    input_size = config['input_size']

    for layer_info in config['layers']:
        if layer_info['type'] == 'linear':
            layers.append(nn.Linear(input_size, layer_info['size']))
            input_size = layer_info['size']
        elif layer_info['type'] == 'relu':
            layers.append(nn.ReLU())

    layers.append(nn.Linear(input_size, config['num_classes']))
    model = nn.Sequential(*layers)

    return model


# Функция для создания конфигураций моделей разной глубины
def create_configs():
    configs = {
        '1_layer': {
            'input_size': 784,
            'num_classes': 10,
            'layers': []
        },
        '2_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'}
            ]
        },
        '3_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'}
            ]
        },
        '5_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'}
            ]
        },
        '7_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'}
            ]
        }
    }

    # Сохраняем конфиги в файлы
    for name, config in configs.items():
        with open(f'{name}_config.json', 'w') as f:
            json.dump(config, f)

    return configs


# Функция для обучения модели
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    # Изменяем размер входных данных перед передачей в модель
    def reshape_input(x):
        return x.view(x.size(0), -1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accs = []
    test_accs = []
    times = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = reshape_input(inputs)  # Изменяем размерность входных данных
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Оценка на тестовом наборе
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = reshape_input(inputs)  # Изменяем размерность входных данных
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        test_accs.append(test_acc)

        epoch_time = time.time() - start_time
        times.append(epoch_time)

        print(
            f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_time:.2f}s')

    return train_losses, train_accs, test_accs, times


# Создаем конфигурации
configs = create_configs()

# Обучаем и оцениваем модели
results = {}
epochs = 15

for name in configs.keys():
    print(f"\nTraining {name} model...")
    config_path = f'{name}_config.json'
    model = create_model_from_config(config_path)

    train_losses, train_accs, test_accs, times = train_model(
        model, train_loader, test_loader, epochs=epochs
    )

    results[name] = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'times': times,
        'avg_time_per_epoch': sum(times) / len(times)
    }

# Визуализация результатов
plt.figure(figsize=(15, 10))

# Графики точности
plt.subplot(2, 2, 1)
for name, res in results.items():
    plt.plot(res['train_accs'], label=f'{name} (train)')
plt.title('Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
for name, res in results.items():
    plt.plot(res['test_accs'], label=f'{name} (test)')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Графики потерь
plt.subplot(2, 2, 3)
for name, res in results.items():
    plt.plot(res['train_losses'], label=name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Время обучения
plt.subplot(2, 2, 4)
avg_times = [res['avg_time_per_epoch'] for res in results.values()]
plt.bar(results.keys(), avg_times)
plt.title('Average Time per Epoch')
plt.ylabel('Seconds')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Вывод итоговых метрик
print("\nFinal Metrics:")
for name, res in results.items():
    print(f"{name}:")
    print(f"  Final Train Accuracy: {res['train_accs'][-1]:.4f}")
    print(f"  Final Test Accuracy: {res['test_accs'][-1]:.4f}")
    print(f"  Average Time per Epoch: {res['avg_time_per_epoch']:.2f}s")
    print(f"  Total Training Time: {sum(res['times']):.2f}s")
    print()

# Задание 1.2

# Создаем конфигурации с регуляризацией
def create_regularized_configs():
    configs = {
        '3_layers_dropout': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.5},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.3}
            ]
        },
        '5_layers_batchnorm': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'batch_norm'},
                {'type': 'relu'}
            ]
        },
        '7_layers_mixed': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 512},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 256},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 256},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'}
            ]
        }
    }

    for name, config in configs.items():
        with open(f'{name}_config.json', 'w') as f:
            json.dump(config, f)

    return configs


# Обучаем и оцениваем модели с регуляризацией
reg_configs = create_regularized_configs()
reg_results = {}

for name in reg_configs.keys():
    print(f"\nTraining {name} model...")
    config_path = f'{name}_config.json'
    model = create_model_from_config(config_path)

    train_losses, train_accs, test_accs, times = train_model(
        model, train_loader, test_loader, epochs=epochs
    )

    reg_results[name] = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'times': times,
        'avg_time_per_epoch': sum(times) / len(times)
    }

# Сравнение моделей с регуляризацией и без
plt.figure(figsize=(15, 5))

# Сравнение test accuracy
plt.subplot(1, 2, 1)
for name in ['3_layers', '3_layers_dropout']:
    if name in results:
        plt.plot(results[name]['test_accs'], label=name)
    elif name in reg_results:
        plt.plot(reg_results[name]['test_accs'], label=name)
plt.title('3 Layers vs 3 Layers with Dropout (Test Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for name in ['5_layers', '5_layers_batchnorm']:
    if name in results:
        plt.plot(results[name]['test_accs'], label=name)
    elif name in reg_results:
        plt.plot(reg_results[name]['test_accs'], label=name)
plt.title('5 Layers vs 5 Layers with BatchNorm (Test Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Анализ переобучения
print("\nOverfitting Analysis:")
for name in ['3_layers', '5_layers', '7_layers']:
    if name in results:
        train_acc = results[name]['train_accs'][-1]
        test_acc = results[name]['test_accs'][-1]
        gap = train_acc - test_acc
        print(f"{name}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, Gap {gap:.4f}")

for name in reg_results.keys():
    train_acc = reg_results[name]['train_accs'][-1]
    test_acc = reg_results[name]['test_accs'][-1]
    gap = train_acc - test_acc
    print(f"{name}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, Gap {gap:.4f}")