# ============================================================
# ЛОКАЛЬНОЕ ОБУЧЕНИЕ YOLO11n на RTX 5070 (8 GB VRAM)
# Windows + multiprocessing-safe
# ============================================================
from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import torch


# ---- CALLBACK ----
def save_checkpoint(trainer):
    src = Path(trainer.save_dir) / 'weights' / 'last.pt'
    if src.exists():
        dst = Path(r'C:\Drone\runs\checkpoints') / f'epoch_{trainer.epoch + 1:03d}.pt'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"  → Checkpoint: {dst.name}")


# ---- ВСЁ ВНУТРИ if __name__ ----
if __name__ == '__main__':
    # Настройка multiprocessing для Windows
    torch.multiprocessing.freeze_support()

    # ---- ПРОВЕРКА ОКРУЖЕНИЯ ----
    print("=" * 60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- ПУТИ ----
    YAML_PATH = r'C:\Drone\yolo_dataset\yolo_dataset\anti_uav.yaml'
    PROJECT_DIR = r'C:\Drone\runs'
    RUN_NAME = 'anti_uav_yolo11n_v3_local'

    assert Path(YAML_PATH).exists(), f"❌ YAML не найден: {YAML_PATH}"
    print(f"✓ YAML: {YAML_PATH}")

    # ---- ОБУЧЕНИЕ ----
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ YOLO11n (RTX 5070, 8 GB VRAM)")
    print("=" * 60)

    model = YOLO('yolo11n.pt')
    model.add_callback('on_train_epoch_end', save_checkpoint)

    start = time.time()

    results = model.train(
        data=YAML_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        optimizer='SGD',
        lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=2.0,
        mosaic=1.0, mixup=0.05, copy_paste=0.2,
        scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        erasing=0.3, fliplr=0.5,
        close_mosaic=5,
        patience=7,
        cache=False,            # ← КРИТИЧНО: никакого 165 GB кэша
        workers=4,
        amp=True,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save_period=5,
        seed=42,
    )

    elapsed = (time.time() - start) / 3600
    print("\n" + "=" * 60)
    print(f"✓ Обучение завершено за {elapsed:.2f} ч")
    print(f"✓ Лучшая модель: {results.save_dir}/weights/best.pt")
    print("=" * 60)