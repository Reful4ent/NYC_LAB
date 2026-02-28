# Инструкция по установке для работы с GPU

## Шаг 1: Активация виртуального окружения

```bash
cd /home/rfflgnt/PycharmProjects/NYC_LAB_1
source venv/bin/activate
```

## Шаг 2: Установка PyTorch с поддержкой CUDA

Ваша система имеет CUDA 12.2, поэтому используйте версию для CUDA 12.1 (совместима):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Или для последней версии CUDA 12.4:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Шаг 3: Установка остальных зависимостей

```bash
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 jupyter
```

## Шаг 4: Проверка установки

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести:
- PyTorch версию
- `CUDA available: True`
- Название вашей GPU (NVIDIA GeForce RTX 4050)

## Шаг 5: Запуск Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

Или в Cursor/VS Code просто откройте `main.ipynb` и выберите интерпретатор из `venv/bin/python`.

## Оптимизации для GPU

Код уже настроен на:
- ✅ Автоматическое определение GPU
- ✅ Batch size 256 для GPU (64 для CPU)
- ✅ Pin memory для ускорения передачи данных
- ✅ Мониторинг использования памяти GPU
- ✅ Многопоточная загрузка данных (num_workers=2)

## Если GPU не определяется

1. Проверьте драйверы: `nvidia-smi`
2. Убедитесь, что установлена версия PyTorch с CUDA
3. Перезапустите Jupyter kernel после установки













