import os
import shutil

# Папка, которую ты распаковал
source_dir = "PKLotSegmented"
# Папка, куда мы всё сложим
target_dir = "raw_data"

os.makedirs(os.path.join(target_dir, "Empty"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "Occupied"), exist_ok=True)

copied_empty = 0
copied_occupied = 0

print("Начинаем сбор картинок из подпапок... Это займет пару минут.")

# Проходимся по всем вложенным папкам
for root, dirs, files in os.walk(source_dir):
    folder_name = os.path.basename(root)
    
    # Если наткнулись на папку с пустыми местами
    if folder_name == "Empty":
        for file in files:
            if file.endswith(".jpg"):
                src = os.path.join(root, file)
                # Переименовываем, чтобы имена не совпали
                dst = os.path.join(target_dir, "Empty", f"{copied_empty}_{file}")
                # Используем move вместо copy, чтобы не занимать лишнее место на диске
                shutil.move(src, dst)
                copied_empty += 1
                
    # Если наткнулись на папку с занятыми местами
    elif folder_name == "Occupied":
        for file in files:
            if file.endswith(".jpg"):
                src = os.path.join(root, file)
                dst = os.path.join(target_dir, "Occupied", f"{copied_occupied}_{file}")
                shutil.move(src, dst)
                copied_occupied += 1

print(f"\nСбор завершен!")
print(f"Пустых мест (Empty): {copied_empty}")
print(f"Занятых мест (Occupied): {copied_occupied}")