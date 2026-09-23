from pathlib import Path
from ultralytics import YOLO

BEST = Path(r"C:\Drone\runs\anti_uav_yolo11n_v3_local\weights\best.pt")
YAML = r"C:\Drone\yolo_dataset\yolo_dataset\anti_uav.yaml"
OUT = Path(r"C:\Drone\android\app\src\main\assets")


def main() -> None:
    model = YOLO(str(BEST))
    print("Export TFLite INT8 imgsz=640 ...")
    path = model.export(
        format="tflite",
        imgsz=640,
        int8=True,
        data=YAML,
        nms=False,
        keras=False,
    )
    print("exported", path)
    src = Path(path)
    dst = OUT / "anti_uav_v3_int8.tflite"
    dst.write_bytes(src.read_bytes())
    print("copied", dst, dst.stat().st_size)


if __name__ == "__main__":
    main()
