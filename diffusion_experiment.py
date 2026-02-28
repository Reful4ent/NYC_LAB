"""
Вариант 5. Диффузионные модели (Stable Diffusion).

Эксперименты с промптом, количеством шагов и семплером.
Требуется: pip install diffusers transformers accelerate
"""

from __future__ import annotations

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

# Подключаем модель Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"

# Загружаем пайплайн (float16 на GPU, float32 на CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

# Изменённый промпт и имя файла
prompt = "A serene mountain landscape at sunset, snow peaks, golden light, photorealistic, 8k"
output_filename = "mountain_sunset.png"

# Семплер Euler вместо DDPMScheduler — даёт другой характер генерации
scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

# Увеличенное количество шагов для более детальной картинки (было 30)
num_inference_steps = 50

print(f"Генерация: {prompt}")
print(f"Семплер: EulerDiscreteScheduler, шагов: {num_inference_steps}, устройство: {device}")

# Генерируем изображение
image = pipe(
    prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.5,
).images[0]

image.save(output_filename)
print(f"Изображение сохранено как '{output_filename}'")
