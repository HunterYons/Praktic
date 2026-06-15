from torchvision import models
import ssl

# На всякий случай отключаем проверку сертификатов, если провайдер шалит
ssl._create_default_https_context = ssl._create_unverified_context

print("Начинаю загрузку весов в кэш...")

print("1. Качаем ResNet18...")
models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print("2. Качаем MobileNetV2...")
models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

print("3. Качаем EfficientNetB0...")
models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

print("4. Качаем SqueezeNet...")
models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)

print("\nУСПЕХ! Все модели в кэше. Интернет для обучения больше не нужен!")