import os
import sys
import torch
import nobuco
from nobuco import ChannelOrder
sys.path.append('../')
from music.mobilenetv4 import mobilenetv4_conv_small_grayscale
from music.globals import DANCES_LABELS

# Load your trained PyTorch model
model_path = "models/mobilenetv4_dance_classifier_noised_v2.pth"
num_classes = len(DANCES_LABELS)

model = mobilenetv4_conv_small_grayscale(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define dummy input for tracing
dummy_input = torch.rand(1, 1, 128, 862)  # [batch, channels, height, width]

# Convert to Keras/TensorFlow model
keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_input],
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
)

# Save the Keras model
keras_model.save("convert/mobilenetv4_dance_classifier.h5")