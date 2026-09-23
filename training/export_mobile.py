"""Export trained best.pt to ONNX for the Android app."""
from pathlib import Path
import shutil
from ultralytics import YOLO

BEST_PT = Path(r"C:\Drone\runs\anti_uav_yolo11n_v3_local\weights\best.pt")
EXPORT_DIR = Path(r"C:\Drone\exports")
ASSETS_DIR = Path(r"C:\Drone\android\app\src\main\assets")


def main() -> None:
    assert BEST_PT.exists(), f"best.pt not found: {BEST_PT}"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading", BEST_PT)
    model = YOLO(str(BEST_PT))
    print("Exporting ONNX imgsz=640 ...")
    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        nms=False,
    )
    src = Path(exported)
    print("Exported:", src)

    for dest in (EXPORT_DIR / "best.onnx", ASSETS_DIR / "best.onnx"):
        shutil.copy(src, dest)
        print("Copied", dest, f"{dest.stat().st_size / 1024**2:.2f} MB")

    pt_copy = EXPORT_DIR / "best.pt"
    shutil.copy(BEST_PT, pt_copy)
    print("Copied source weights", pt_copy)


if __name__ == "__main__":
    main()
