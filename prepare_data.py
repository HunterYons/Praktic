import os
import shutil
import random

# Настройки путей
source_dir = "raw_data"
base_dir = "data"
classes = ["Empty", "Occupied"]
split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}

# ЛИМИТ: берем только по 10 000 случайных картинок каждого класса
MAX_IMAGES_PER_CLASS = 10000 

# Создаем нужные папки (train, val, test)
for split in split_ratio.keys():
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

print("Начинаем перемешивание и распределение (берем подвыборку)...")

for cls in classes:
    src_cls_dir = os.path.join(source_dir, cls)
    
    if not os.path.exists(src_cls_dir):
        print(f"Ошибка: Не найдена папка {src_cls_dir}!")
        continue

    # Получаем список всех картинок в папке класса
    images = os.listdir(src_cls_dir)
    
    # Перемешиваем список, чтобы выборка была случайной
    random.shuffle(images) 

    # Обрезаем список до нашего лимита
    images = images[:MAX_IMAGES_PER_CLASS]

    # Считаем индексы для срезов
    train_split = int(len(images) * split_ratio["train"])
    val_split = int(len(images) * (split_ratio["train"] + split_ratio["val"]))

    # Разрезаем список на три части
    train_imgs = images[:train_split]
    val_imgs = images[train_split:val_split]
    test_imgs = images[val_split:]

    print(f"Класс {cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    # Копируем файлы по новым папкам
    for img_list, split in zip([train_imgs, val_imgs, test_imgs], ["train", "val", "test"]):
        for img in img_list:
            src_path = os.path.join(src_cls_dir, img)
            dst_path = os.path.join(base_dir, split, cls, img)
            shutil.copy(src_path, dst_path)

print("\nУспех! Сбалансированный мини-датасет сформирован в папке 'data'.")