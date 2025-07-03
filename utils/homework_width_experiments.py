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