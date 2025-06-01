from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import os

# 模型加载Model loading（仅加载一次）
model_path = "app/model/dog_breed"
model = ViTForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
model.eval()

def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]
