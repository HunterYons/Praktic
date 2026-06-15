import torch

print("PyTorch успешно инициализирован!")
print("Доступен ли CUDA (GPU):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Имя твоей видеокарты:", torch.cuda.get_device_name(0))