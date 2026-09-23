"""
Оценка v4 на Anti-UAV test split.
Windows-safe: обёрнуто в if __name__ == '__main__' + freeze_support().
"""
from ultralytics import YOLO
import torch
from pathlib import Path


if __name__ == '__main__':
    # КРИТИЧНО для Windows: правильный spawn процессов
    torch.multiprocessing.freeze_support()

    # ---- ПУТИ (raw strings, чтобы не было escape-предупреждений) ----
    MODEL_PATH = Path(r'C:\Drone\runs\anti_uav_yolo11n_v4_combined\weights\best.pt')
    YAML_PATH = Path(r'C:\Drone\yolo_dataset\yolo_dataset\anti_uav.yaml')

    # ---- ПРОВЕРКА ----
    assert MODEL_PATH.exists(), f"❌ Модель не найдена: {MODEL_PATH}"
    assert YAML_PATH.exists(), f"❌ YAML не найден: {YAML_PATH}"

    print("=" * 60)
    print("ОЦЕНКА V4 НА ANTI-UAV TEST SPLIT")
    print("=" * 60)
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {YAML_PATH}")
    print("=" * 60)

    # ---- ЗАГРУЗКА МОДЕЛИ ----
    model = YOLO(str(MODEL_PATH))

    # ---- ВАЛИДАЦИЯ ----
    metrics = model.val(
        data=str(YAML_PATH),
        split='test',                # test = hold-out, не использовался в обучении
        imgsz=640,
        batch=16,
        workers=0,                   # 0 = без subprocess, обходит проблему Windows
        project=r'C:\Drone\runs',
        name='v4_eval_anti_uav_test',
        plots=True,
        verbose=True,
    )

    # ---- РЕЗУЛЬТАТЫ ----
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ НА ANTI-UAV TEST")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("=" * 60)

    # ---- СРАВНЕНИЕ С V3 ----
    print("\nСравнение с v3 (обучалась только на Anti-UAV):")
    print(f"  v3 mAP50:  0.994")
    print(f"  v4 mAP50:  {metrics.box.map50:.4f}")
    delta = metrics.box.map50 - 0.994
    print(f"  Разница:   {delta:+.4f}")

    if metrics.box.map50 >= 0.97:
        print("\n✅ v4 сохранила знания Anti-UAV — можно экспортировать")
    elif metrics.box.map50 >= 0.93:
        print("\n✅ Приемлемо — экспортируем как есть")
    else:
        print("\n⚠️ Заметное падение — обсудим дообучение")