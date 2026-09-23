# ============================================================
# ЭКСПОРТ МОДЕЛИ В TFLite INT8 (для Android)
# ============================================================
from ultralytics import YOLO
from pathlib import Path
import time
import shutil

# ---- ПУТИ ----
BEST_PT = r'C:\Drone\runs\anti_uav_yolo11n_v3_local\weights\best.pt'
YAML_PATH = r'C:\Drone\yolo_dataset\yolo_dataset\anti_uav.yaml'
EXPORT_DIR = Path(r'C:\Drone\exports')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    print("=" * 60)
    print("ЭКСПОРТ YOLO11n В МОБИЛЬНЫЕ ФОРМАТЫ")
    print("=" * 60)
    print(f"Источник: {BEST_PT}")
    print(f"Размер: {Path(BEST_PT).stat().st_size / 1024**2:.2f} MB")

    model = YOLO(BEST_PT)

    # ---- 1. TFLite INT8 (основной артефакт для Android) ----
    print("\n[1/3] Экспорт в TFLite INT8...")
    print("  Калибровка на val-датасете — займёт 3-7 минут")
    start = time.time()
    int8_path = model.export(
        format='litert',
        imgsz=640,
        quantize=8,               # INT8 квантование
        data=YAML_PATH,           # Калибровка на реальных изображениях
    )
    print(f"  ✓ Готово за {(time.time()-start)/60:.1f} мин: {int8_path}")

    # ---- 2. TFLite FP32 (запасной, для отладки) ----
    print("\n[2/3] Экспорт в TFLite FP32...")
    start = time.time()
    fp32_path = model.export(
        format='litert',
        imgsz=640,
        quantize=32,
    )
    print(f"  ✓ Готово за {(time.time()-start)/60:.1f} мин: {fp32_path}")

    # ---- 3. ONNX (для отладки и десктопа) ----
    print("\n[3/3] Экспорт в ONNX...")
    start = time.time()
    onnx_path = model.export(
        format='onnx',
        imgsz=640,
        opset=12,
        simplify=True,
    )
    print(f"  ✓ Готово за {(time.time()-start)/60:.1f} мин: {onnx_path}")

    # ---- КОПИРУЕМ ВСЁ В ОДНУ ПАПКУ ----
    print("\n" + "=" * 60)
    print("СБОР РЕЗУЛЬТАТОВ")
    print("=" * 60)

    weights_dir = Path(BEST_PT).parent
    for fname in ['best.pt', 'best_int8.tflite', 'best.tflite', 'best.onnx']:
        src = weights_dir / fname
        if src.exists():
            dst = EXPORT_DIR / fname
            shutil.copy(src, dst)
            size_mb = dst.stat().st_size / 1024**2
            print(f"  {fname:30s} {size_mb:>8.2f} MB  →  {dst}")

    print("\n" + "=" * 60)
    print(f"✓ ВСЁ ГОТОВО. Папка: {EXPORT_DIR}")
    print("=" * 60)
    print("\nЧто передать разработчику Android:")
    print(f"  → {EXPORT_DIR}\\best_int8.tflite (основная модель)")
    print("\nЧто оставить себе:")
    print(f"  → {EXPORT_DIR}\\best.pt (для дальнейшего дообучения)")