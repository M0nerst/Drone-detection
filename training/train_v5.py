from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import torch

# ---- КОНФИГ ----
YOLO_START = r'C:\Drone\runs\anti_uav_yolo11n_v4_combined\weights\best.pt'
YAML_PATH  = r'C:\Drone\combined_dataset_v5\anti_uav_combined_v5.yaml'
PROJECT = r'C:\Drone\runs'
RUN_NAME   = 'anti_uav_yolo11n_v5_negatives'
BACKUP = Path(r'C:\Drone\runs\checkpoints_v5')
BACKUP.mkdir(parents=True, exist_ok=True)


def save_checkpoint(trainer):
    src = Path(trainer.save_dir) / 'weights' / 'last.pt'
    if src.exists():
        shutil.copy(src, BACKUP / f'epoch_{trainer.epoch + 1:03d}.pt')


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    assert Path(YOLO_START).exists(), f"❌ Не найдена v4: {YOLO_START}"
    assert Path(YAML_PATH).exists(), f"❌ Не найден YAML: {YAML_PATH}"

    print("=" * 60)
    print("ДООБУЧЕНИЕ v5 (v4 + hard negatives)")
    print("=" * 60)

    model = YOLO(YOLO_START)
    model.add_callback('on_train_epoch_end', save_checkpoint)

    start = time.time()

    results = model.train(
        data=YAML_PATH,
        epochs=8,                  # 8 эпох — дообучение, не с нуля
        imgsz=640,
        batch=8,
        optimizer='SGD',
        lr0=0.001,                 # маленький LR — не разрушаем v4
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.2,
        scale=0.5,
        translate=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        erasing=0.3,
        fliplr=0.5,
        close_mosaic=3,
        patience=5,
        cache=False,
        workers=4,
        amp=True,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save_period=4,
        seed=42,
    )

    elapsed = (time.time() - start) / 3600
    print(f"\n✓ Обучение за {elapsed:.2f} ч")
    print(f"✓ Модель: {results.save_dir}/weights/best.pt")