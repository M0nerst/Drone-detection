from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import torch

# ---- КОНФИГ ----
YOLO_START = r'C:\Drone\runs\anti_uav_yolo11n_v3_local\weights\best.pt'
YAML_PATH = r'C:\Drone\combined_dataset\anti_uav_combined.yaml'
PROJECT = r'C:\Drone\runs'
RUN_NAME = 'anti_uav_yolo11n_v4_combined'
BACKUP = Path(r'C:\Drone\runs\checkpoints_v4')
BACKUP.mkdir(parents=True, exist_ok=True)


def save_checkpoint(trainer):
    src = Path(trainer.save_dir) / 'weights' / 'last.pt'
    if src.exists():
        shutil.copy(src, BACKUP / f'epoch_{trainer.epoch + 1:03d}.pt')


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    assert Path(YOLO_START).exists(), f"❌ Не найдена v3: {YOLO_START}"
    assert Path(YAML_PATH).exists(), f"❌ Не найден YAML: {YAML_PATH}"

    print("=" * 60)
    print("ДООБУЧЕНИЕ YOLO11n v4 (Combined Dataset)")
    print("=" * 60)
    print(f"Стартуем с: {YOLO_START}")
    print(f"Датасет:    {YAML_PATH}")
    print(f"Train: 52,576 | Val: 8,421")
    print(f"Эпох: 20 | Batch: 8 | imgsz: 640")
    print("=" * 60)

    model = YOLO(YOLO_START)
    model.add_callback('on_train_epoch_end', save_checkpoint)

    start = time.time()

    results = model.train(
        data=YAML_PATH,
        epochs=10,
        imgsz=640,
        batch=16,
        optimizer='SGD',
        lr0=0.002,              # маленький LR — не разрушаем Anti-UAV
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        scale=0.5,
        translate=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        erasing=0.4,
        fliplr=0.5,
        close_mosaic=5,
        patience=5,
        cache=False,
        workers=4,
        amp=True,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save_period=5,
        seed=42,
    )

    elapsed = (time.time() - start) / 3600
    print(f"\n✓ Обучение завершено за {elapsed:.2f} ч")
    print(f"✓ Модель: {results.save_dir}/weights/best.pt")