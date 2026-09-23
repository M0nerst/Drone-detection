# Drone Detection — обнаружение дронов в реальном времени

Система компьютерного зрения для обнаружения дронов (БПЛА) в реальном времени.
Модель YOLO11n обучена на расширенном датасете (Anti-UAV + Roboflow + hard negatives)
и оптимизирована для мобильных устройств и edge-платформ.

## 📊 Итоговые метрики

| Метрика | Значение |
|---|---|
| mAP50 (Anti-UAV test) | **0.975** |
| mAP50-95 | 0.557 |
| Precision | 0.974 |
| Recall | 0.942 |
| Размер модели (INT8) | 2.9 MB |
| FPS (Snapdragon 8 Gen 2, NNAPI) | 25-40 |
| FPS (Snapdragon 778G, GPU) | 12-18 |

## 🎯 Возможности

- ✅ Детекция дронов на фоне неба (Anti-UAV)
- ✅ Детекция дронов крупным планом (Roboflow)
- ✅ Устойчивость к ложным срабатываниям (11K hard negatives)
- ✅ Работа в реальном времени на смартфоне
- ⏳ Развёртывание на edge-платформах (Repka Pi 5, Jetson)

## 🏗 Архитектура

```
┌─────────────────────────────────────────┐
│  CameraX → ImageProxy                   │
│         ↓                               │
│  Letterbox (640×640, сохранить пропорции)│
│         ↓                               │
│  TFLite Interpreter (INT8, NNAPI/GPU)  │
│         ↓                               │
│  Output: [1, 5, 8400]                   │
│         ↓                               │
│  NMS на Kotlin (IoU=0.45, conf=0.5)    │
│         ↓                               │
│  Отрисовка bbox на OverlayView          │
└─────────────────────────────────────────┘
```

## 📦 Структура проекта

```
Drone-detection/
├── training/        # Скрипты обучения и экспорта модели
├── android/         # Исходники Android-приложения (Kotlin)
├── models/          # Финальные .tflite-модели
├── docs/            # Документация
└── README.md
```

## 🚀 Быстрый старт

### Требования

- Python 3.10+
- CUDA-совместимая GPU (для обучения) или CPU (для инференса)
- Android Studio Hedgehog+ (для сборки приложения)

### Установка

```bash
git clone https://github.com/M0nerst/Drone-detection.git
cd drone-detection
pip install -r requirements.txt
```

### Использование готовой модели

Модель `models/anti_uav_v5_int8.tflite` (2.9 MB) готова к использованию:

```python
from ultralytics import YOLO

model = YOLO('models/anti_uav_v5_int8.tflite')
results = model.predict('drone.jpg', conf=0.5)
results[0].show()
```

### Обучение с нуля (или дообучение)

1. **Скачайте датасеты:**
   - [Anti-UAV300](https://github.com/ZhaoJ9014/Anti-UAV) (требует запроса)
   - [Roboflow: drone detection](https://universe.roboflow.com/)
   - [Roboflow: UAV Detection](https://universe.roboflow.com/)
   - [Roboflow: Quadcopter](https://universe.roboflow.com/)
   - [Roboflow: bird vs drone](https://universe.roboflow.com/)

2. **Соберите датасет:**
   ```bash
   python training/convert_anti_uav.py      # Anti-UAV → YOLO
   python training/build_combined_dataset.py # + Roboflow
   python training/merge_v5.py              # + hard negatives
   ```

3. **Обучите модель:**
   ```bash
   python training/train_v5.py
   ```
   ~1.5 часа на RTX 5070.

4. **Экспортируйте в INT8:**
   ```bash
   python training/export_int8.py
   ```
   Получите `anti_uav_v5_int8.tflite` (2.9 MB).

### Сборка Android-приложения

```bash
cd android
./gradlew assembleRelease
```

APK появится в `android/app/build/outputs/apk/release/`.

## 📚 Документация

- [`docs/dataset.md`](docs/dataset.md) — описание датасетов и подготовка
- [`docs/training.md`](docs/training.md) — детали обучения и гиперпараметры
- [`docs/android_notes.md`](docs/android_notes.md) — заметки по интеграции модели

## 📊 Датасет

| Компонент | Изображений | Назначение |
|---|---|---|
| Anti-UAV300 | 33,438 | Дроны в небе |
| Roboflow (4 датасета) | 27,559 | Дроны крупным планом |
| Hard negatives (CW + SD) | 10,930 | Фон, дороги, знаки |
| **ИТОГО** | **~62,000** | Train: 61,376 / Val: 9,399 |

**Распределение классов:** `0: drone` (один класс, детекция)

## 🔬 Технические детали

### Модель
- **Архитектура:** YOLO11n (Ultralytics)
- **Параметров:** 2,582,347
- **Вход:** `[1, 640, 640, 3]` float32, NHWC, [0, 1]
- **Выход:** `[1, 5, 8400]` — x_center, y_center, w, h, confidence
- **Квантование:** INT8 (weights + activations)

### Обучение
- **Стартовая модель:** YOLO11n COCO
- **Optimizer:** SGD, lr0=0.001
- **Batch:** 8, imgsz: 640
- **Epochs:** 8 (v5, дообучение с v4)
- **Аугментации:** mosaic, mixup, copy-paste, hsv, erasing

### Инференс на Android
- **LiteRT** (бывший TensorFlow Lite) 1.4.0
- **Делегаты:** NNAPI (приоритет), GPU, XNNPACK (fallback)
- **NMS:** реализован на Kotlin (модель не содержит встроенного NMS)

## 🛣 Дорожная карта

- [x] Датасет (Anti-UAV + Roboflow)
- [x] Обучение v3 (Anti-UAV only)
- [x] Обучение v4 (combined dataset)
- [x] Обучение v5 (+ hard negatives)
- [x] Экспорт INT8
- [x] Android-приложение
- [ ] Развёртывание на Repka Pi 5 / Jetson Orin Nano
- [ ] Интеграция с полётным контроллером (MAVLink)
- [ ] Трекинг (ByteTrack) для стабилизации детекций

## ⚠️ Ограничения

- Модель обучена на данных дневного и ИК-диапазона. В **тёмное время суток** могут быть ложные срабатывания на яркие объекты (фонари, экраны).
- **Ночные сцены** требуют дополнительного обучения (см. roadmap).
- Для управления дроном требуется фильтрация + трекинг (в разработке).

## 📄 Лицензия

MIT License — см. [LICENSE](LICENSE).

## 🙏 Благодарности

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO framework
- [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) — базовый датасет
- [Roboflow Universe](https://universe.roboflow.com/) — дополнительные датасеты
- Google AI Edge — LiteRT (TFLite)

## 📬 Контакты

Проект создан для личного использования. Вопросы — через GitHub Issues.