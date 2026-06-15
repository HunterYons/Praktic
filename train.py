import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time

# --- НАСТРОЙКИ ---
EPOCHS = 3
BATCH_SIZE = 32
DATA_DIR = 'data'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Устройство (Используем твою RTX 5070)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение будет проходить на: {device}")

# --- ПОДГОТОВКА ДАННЫХ ---
# Нейросети любят стандартный размер картинки 224x224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Легкая аугментация
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Классы: {class_names}. Размер train: {dataset_sizes['train']}, val: {dataset_sizes['val']}")

# --- 5 РАЗНЫХ АРХИТЕКТУР ---

# 1. Своя простая сверточная сеть (Кастомная)
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2) # 2 класса

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(name):
    if name == "CustomCNN":
        model = CustomCNN()
    elif name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, 2)
    elif name == "SqueezeNet":
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
        model.num_classes = 2
    
    return model.to(device)

# --- ФУНКЦИЯ ОБУЧЕНИЯ ---
def train_model(model, name, criterion, optimizer, num_epochs=EPOCHS):
    print(f"\n{'='*40}\nНачинаем обучение модели: {name}\n{'='*40}")
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Эпоха {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Сохраняем веса лучшей эпохи
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{name}.pth'))

    time_elapsed = time.time() - since
    print(f'\nОбучение {name} завершено за {time_elapsed // 60:.0f}м {time_elapsed % 60:.0f}с')
    print(f'Лучшая точность на валидации: {best_acc:4f}')
    return best_acc

# --- ЗАПУСК КОНВЕЙЕРА ---
if __name__ == '__main__':
    models_to_train = ["CustomCNN", "ResNet18", "MobileNetV2", "EfficientNetB0", "SqueezeNet"]
    results = {}

    criterion = nn.CrossEntropyLoss()

    for model_name in models_to_train:
        model = get_model(model_name)
        # Оптимизатор Adam обычно показывает лучшие результаты
        optimizer = optim.Adam(model.parameters(), lr=0.001) 
        
        best_accuracy = train_model(model, model_name, criterion, optimizer, num_epochs=EPOCHS)
        results[model_name] = best_accuracy

    print("\n\n" + "*"*40)
    print("ИТОГОВЫЙ РЕЙТИНГ МОДЕЛЕЙ (Точность):")
    print("*"*40)
    # Сортируем результаты от лучшего к худшему
    for name, acc in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {acc*100:.2f}%")