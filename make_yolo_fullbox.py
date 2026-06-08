from pathlib import Path
import shutil

src_root = Path("dataset")
dst_root = Path("rfuav_yolo")

classes = ["YunZhuo-H12", "YunZhuo-H16", "YunZhuo-H30"]
class_to_id = {name: i for i, name in enumerate(classes)}

split_map = {
    "train": "train",
    "valid": "val",
}

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

for src_split, yolo_split in split_map.items():
    for cls_name in classes:
        src_dir = src_root / src_split / cls_name

        if not src_dir.exists():
            print(f"Missing folder: {src_dir}")
            continue

        image_out = dst_root / "images" / yolo_split
        label_out = dst_root / "labels" / yolo_split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() not in image_exts:
                continue

            new_name = f"{cls_name}_{img_path.name}"
            dst_img = image_out / new_name
            dst_label = label_out / f"{Path(new_name).stem}.txt"

            shutil.copy2(img_path, dst_img)

            cls_id = class_to_id[cls_name]

            # Full-image box: class x_center y_center width height
            dst_label.write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n")

print("Done. YOLO dataset created at:", dst_root)