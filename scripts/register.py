import tlc
from pathlib import Path

# Config
CLASSES = ["chihuahua", "muffin", "undefined"]
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

schemas = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=CLASSES),
    "weight": tlc.SampleWeightSchema(),
}

def register_dataset_to_table(dataset_path: Path, table_name: str, split_name: str, include_undefined: bool = False):
    dataset_path = Path(dataset_path)
    image_data = []
    classes_to_process = CLASSES[:-1] if not include_undefined else CLASSES

    for class_idx, class_name in enumerate(classes_to_process):
        class_folder = dataset_path / class_name
        if class_folder.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_files = sorted(class_folder.glob(ext))
                for img_path in image_files:
                    label = class_idx if class_name != "undefined" else 2
                    image_data.append({"path": str(img_path.absolute()), "label": label})

    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Chihuahua vs Muffin {split_name} set",
        column_schemas=schemas,
        if_exists="overwrite",
    )

    for i, data in enumerate(image_data):
        label = data["label"]
        weight = 1.0 if label in [0, 1] else 0.0
        table_writer.add_row({
            "id": i,
            "image": data["path"],
            "label": label,
            "weight": weight,
        })

    return table_writer.finalize()

def main():
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data"

    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return

    tlc.register_project_url_alias(
        token="CHIHUAHUA_MUFFIN_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )

    register_dataset_to_table(data_path / "train", "train", "train", include_undefined=True)
    register_dataset_to_table(data_path / "val", "val", "val", include_undefined=False)
    print("Successfully registered tables!")

if __name__ == "__main__":
    main()
