from ultralytics import YOLO
from pathlib import Path
import torch

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    MODEL = Path(r'C:\Drone\runs\anti_uav_yolo11n_v5_negatives\weights\best.pt')
    YAML  = Path(r'C:\Drone\yolo_dataset\yolo_dataset\anti_uav.yaml')

    assert MODEL.exists(), f"❌ Модель не найдена: {MODEL}"
    assert YAML.exists(), f"❌ YAML не найден: {YAML}"

    model = YOLO(str(MODEL))
    metrics = model.val(
        data=str(YAML),
        split='test',
        imgsz=640,
        batch=16,
        workers=0,
        project=r'C:\Drone\runs',
        name='v5_eval_anti_uav_test',
    )

    print("\n" + "=" * 60)
    print("V5 НА ANTI-UAV TEST (чистый домен)")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print()
    print("Сравнение с v4:")
    print(f"  v4 на Anti-UAV test: mAP50=0.9769, P=0.9727, R=0.9475")
    print(f"  v5 на Anti-UAV test: mAP50={metrics.box.map50:.4f}, P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}")

    if metrics.box.mr < 0.85:
        print("\n⚠️ RECALL < 0.85 — надо уменьшить долю негативов и переобучить")
    elif metrics.box.mr < 0.90:
        print("\n⚠️ Recall 0.85-0.90 — приемлемо, но можно улучшить")
    else:
        print("\n✅ RECALL ≥ 0.90 — отлично, можно экспортировать в INT8")