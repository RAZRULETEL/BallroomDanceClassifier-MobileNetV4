import torch

from music.mobilenetv4 import mobilenetv4_conv_small_grayscale

# Load the provided MobileNetV4 implementation
# Instantiate the model with 10 output classes
model = mobilenetv4_conv_small_grayscale(num_classes=10)
model.eval()  # Switch to evaluation mode

input_tensor = torch.randn(1, 1, 128, 862)  # Batch size = 1
output = model(input_tensor)
print(output)  # Should be (1, 10)