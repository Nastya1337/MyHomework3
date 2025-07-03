# Домашнее задание к уроку 3: Полносвязные сети
## Задание 1: Эксперименты с глубиной сети (30 баллов)

##@ 1.1 Сравнение моделей разной глубины (15 баллов)
Создайте и обучите модели с различным количеством слоев:
- 1 слой (линейный классификатор)
- 2 слоя (1 скрытый)
- 3 слоя (2 скрытых)
- 5 слоев (4 скрытых)
- 7 слоев (6 скрытых)

Для каждого варианта:
- Сравните точность на train и test
- Визуализируйте кривые обучения
- Проанализируйте время обучения

### 1.2 Анализ переобучения (15 баллов)
Исследуйте влияние глубины на переобучение:
- Постройте графики train/test accuracy по эпохам
- Определите оптимальную глубину для каждого датасета
- Добавьте Dropout и BatchNorm, сравните результаты
- Проанализируйте, когда начинается переобучение

```python
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
```

Загрузка данных: MNIST загружается и преобразуется в тензоры с нормализацией.
1. Создание моделей:

- Реализована функция create_model_from_config, которая строит модель на основе JSON-конфигурации.
- Создаются конфигурации для моделей разной глубины (1, 2, 3, 5, 7 слоев) с ReLU-активациями.

2. Обучение моделей:

- Используется кросс-энтропия как функция потерь и оптимизатор Adam.
- Логируются метрики (loss, accuracy) на тренировочном и тестовом наборах.

3. Визуализация:

- Строятся графики accuracy и loss для сравнения моделей.
- Анализируется время обучения для разных архитектур.

4. Эксперименты с регуляризацией:

- Добавляются модели с Dropout и BatchNorm слоями.
- Сравнивается их эффективность с базовыми моделями.

5. Анализ переобучения:

- Вычисляется разница между тренировочной и тестовой accuracy.
- Сравнивается влияние регуляризации на переобучение.

Код демонстрирует, как глубина сети и методы регуляризации влияют на качество классификации и склонность к переобучению.

## Задание 2: Эксперименты с шириной сети 

### 2.1 Сравнение моделей разной ширины 
Создайте модели с различной шириной слоев:
- Узкие слои: [64, 32, 16]
- Средние слои: [256, 128, 64]
- Широкие слои: [1024, 512, 256]
- Очень широкие слои: [2048, 1024, 512]

Для каждого варианта:
- Поддерживайте одинаковую глубину (3 слоя)
- Сравните точность и время обучения
- Проанализируйте количество параметров
  
### 2.2 Оптимизация архитектуры 
Найдите оптимальную архитектуру:
- Используйте grid search для поиска лучшей комбинации
- Попробуйте различные схемы изменения ширины (расширение, сужение, постоянная)
- Визуализируйте результаты в виде heatmap

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import tqdm


# Генерация синтетических данных
def generate_data(num_samples=10000, num_features=100, num_classes=10):
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# Класс модели
class MLP(nn.Module):
    def __init__(self, input_size=100, hidden_sizes=[256, 128, 64], output_size=10):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Функция обучения
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Валидация
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(y_batch.numpy())
                y_pred.extend(predicted.numpy())

        val_acc = accuracy_score(y_true, y_pred)
        val_accuracies.append(val_acc)

    training_time = time.time() - start_time
    return train_losses, val_accuracies, training_time


# 2.1 Сравнение моделей разной ширины
def compare_width_models():
    # Конфигурации моделей
    configs = {
        'Узкие': [64, 32, 16],
        'Средние': [256, 128, 64],
        'Широкие': [1024, 512, 256],
        'Очень широкие': [2048, 1024, 512]
    }

    # Генерация данных
    X, y = generate_data()
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    results = []

    for name, hidden_sizes in configs.items():
        print(f"\nТренировка модели: {name} {hidden_sizes}")
        model = MLP(hidden_sizes=hidden_sizes)

        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())

        # Обучение
        train_losses, val_accuracies, training_time = train_model(model, train_loader, val_loader)

        # Сохранение результатов
        results.append({
            'Название': name,
            'Архитектура': hidden_sizes,
            'Параметры': total_params,
            'Время обучения': training_time,
            'Лучшая точность': max(val_accuracies)
        })

        # Визуализация кривых обучения
        plt.plot(val_accuracies, label=f"{name} ({total_params:,} параметров)")

    plt.title('Сравнение точности моделей')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()

    # Вывод таблицы результатов
    import pandas as pd
    df = pd.DataFrame(results)
    print("\nРезультаты сравнения моделей:")
    print(df[['Название', 'Архитектура', 'Параметры', 'Время обучения', 'Лучшая точность']])


# 2.2 Оптимизация архитектуры
def optimize_architecture():
    # Параметры для grid search
    param_grid = {
        'first_layer': [64, 128, 256, 512],
        'second_layer': [64, 128, 256, 512],
        'third_layer': [64, 128, 256, 512],
        'scheme': ['расширение', 'сужение', 'постоянная']
    }

    # Фильтр для схем изменения ширины
    def is_valid_combination(params):
        if params['scheme'] == 'расширение':
            return params['first_layer'] < params['second_layer'] < params['third_layer']
        elif params['scheme'] == 'сужение':
            return params['first_layer'] > params['second_layer'] > params['third_layer']
        else:  # постоянная
            return params['first_layer'] == params['second_layer'] == params['third_layer']

    # Генерация данных
    X, y = generate_data()
    dataset = TensorDataset(X, y)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    results = []
    valid_params = [p for p in ParameterGrid(param_grid) if is_valid_combination(p)]

    for params in tqdm.tqdm(valid_params, desc="Grid Search"):
        hidden_sizes = [params['first_layer'], params['second_layer'], params['third_layer']]
        model = MLP(hidden_sizes=hidden_sizes)

        # Обучение (уменьшим количество эпох для ускорения)
        _, val_accuracies, training_time = train_model(model, train_loader, val_loader, epochs=10)

        results.append({
            'first_layer': params['first_layer'],
            'second_layer': params['second_layer'],
            'third_layer': params['third_layer'],
            'scheme': params['scheme'],
            'accuracy': max(val_accuracies),
            'time': training_time
        })

    # Анализ результатов
    import pandas as pd
    df = pd.DataFrame(results)

    # Визуализация heatmap для точности
    for scheme in ['расширение', 'сужение', 'постоянная']:
        scheme_df = df[df['scheme'] == scheme]
        if len(scheme_df) > 0:
            # Для постоянной схемы нужен особый подход
            if scheme == 'постоянная':
                plt.figure()
                plt.plot(scheme_df['first_layer'], scheme_df['accuracy'])
                plt.title(f'Точность для схемы "{scheme}"')
                plt.xlabel('Размер слоя')
                plt.ylabel('Точность')
            else:
                # Создаем pivot таблицу для heatmap
                pivot_df = scheme_df.pivot_table(index='first_layer',
                                                 columns='second_layer',
                                                 values='accuracy',
                                                 aggfunc='mean')
                plt.figure()
                sns.heatmap(pivot_df, annot=True, fmt=".3f")
                plt.title(f'Точность для схемы "{scheme}"')

    plt.show()

    # Вывод лучших комбинаций
    print("\nЛучшие комбинации параметров:")
    print(df.sort_values('accuracy', ascending=False).head(10))


# Запуск сравнения моделей
print("=== 2.1 Сравнение моделей разной ширины ===")
compare_width_models()

# Запуск оптимизации архитектуры
print("\n=== 2.2 Оптимизация архитектуры ===")
optimize_architecture()
```

Этот код представляет собой исследование влияния ширины (количества нейронов в слоях) и архитектуры (схемы изменения ширины между слоями) на производительность многослойного перцептрона (MLP) при работе с синтетическими данными.

Основные компоненты:

1. Генерация данных:

- Создаются синтетические данные (generate_data()): 10,000 образцов с 100 признаками и 10 классами.

2. Модель MLP:

- Реализован класс MLP с настраиваемой архитектурой (количество и размер скрытых слоев).
- Все модели используют ReLU-активации между линейными слоями.

3. Функция обучения:

- train_model() обучает модель с кросс-энтропийной потерей и оптимизатором Adam.
- Логируются потери на обучении и точность на валидационном наборе.

4. Эксперименты:

Сравнение моделей разной ширины:

1. Тестируются 4 типа архитектур: от узких (64-32-16) до очень широких (2048-1024-512).
2. Визуализируются кривые обучения и сравниваются:

- Количество параметров
- Время обучения
- Лучшая точность

Оптимизация архитектуры:

Проводится grid search по:

- Размерам каждого из 3 слоев (64, 128, 256, 512)
- Схемам изменения ширины:

  - Расширение (размеры слоев увеличиваются)
  - Сужение (размеры слоев уменьшаются)
  - Постоянная (все слои одинакового размера)

Результаты визуализируются с помощью heatmap (для расширяющихся/сужающихся схем) и графиков (для постоянных схем).
Выводятся 10 лучших комбинаций параметров.

5. Визуализация:

- Используются matplotlib и seaborn для графиков и heatmap.
- Результаты представлены в табличном формате с помощью pandas.

Код позволяет наглядно увидеть компромиссы при выборе архитектуры нейронной сети и определить оптимальные конфигурации для заданного типа данных.

## Задание 3: Эксперименты с регуляризацией 

### 3.1 Сравнение техник регуляризации 
Исследуйте различные техники регуляризации:

- Без регуляризации
- Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
- Только BatchNorm
- Dropout + BatchNorm
- L2 регуляризация (weight decay)

Для каждого варианта:
- Используйте одинаковую архитектуру
- Сравните финальную точность
- Проанализируйте стабильность обучения
- Визуализируйте распределение весов
  
### 3.2 Адаптивная регуляризация (10 баллов)
Реализуйте адаптивные техники:
- Dropout с изменяющимся коэффициентом
- BatchNorm с различными momentum
- Комбинирование нескольких техник
- Анализ влияния на разные слои сети

```python
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
```

Этот код представляет собой исследование различных техник регуляризации в сверточных нейронных сетях на примере классификации изображений CIFAR-10. Основное внимание уделяется сравнению стандартных и адаптивных методов регуляризации.

Основные компоненты:

1. Архитектура сети:

- BaseNet - базовая CNN с 2 сверточными и 2 полносвязными слоями
- AdaptiveNet - улучшенная версия с адаптивной регуляризацией

2. Методы регуляризации:

- Dropout (разные коэффициенты)
- Batch Normalization
- L2-регуляризация
- Адаптивный Dropout (с изменением rate во время обучения)
- Layer-specific Dropout (разные коэффициенты для разных слоев)
- Настройка momentum в BatchNorm

3. Эксперименты:

Сравнение техник регуляризации:

Тестируются 7 различных комбинаций регуляризации
Сравниваются их влияние на:

- Кривые обучения
- Финальную точность
- Распределение весов

Адаптивная регуляризация:

Исследуются 5 адаптивных стратегий:

- Фиксированный Dropout
- Возрастающий Dropout (0.1 → 0.5)
- Высокий momentum в BatchNorm
- Layer-specific Dropout
- Комбинированный адаптивный подход

4. Визуализация:

- Графики потерь и точности
- Гистограммы распределения весов
- Таблицы сравнения финальных метрик

Код позволяет наглядно увидеть эффект от разных методов регуляризации и выбрать оптимальную стратегию для конкретной задачи. Особый интерес представляет сравнение статических и адаптивных подходов.

## Вывод
Регуляризация критически важна для устойчивого обучения CNN. Адаптивные методы показывают лучшие результаты, чем статические, так как позволяют гибко управлять обучением на разных этапах. Оптимальный выбор зависит от конкретной задачи, но комбинация нескольких методов (например, адаптивный Dropout + BatchNorm) чаще всего дает наилучший баланс между точностью и устойчивостью модели.
