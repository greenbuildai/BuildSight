from ultralytics import YOLO
import os

def train_yolo(data_yaml_path, epochs=100, img_size=640, model_name="yolo11n.pt"):
    """
    Trains a YOLO model using the specified dataset.
    """
    print(f"Starting training with model {model_name} on {data_yaml_path}")
    
    # Load a model
    model = YOLO(model_name)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        plots=True,
        save=True,
        project="backend/models",
        name="custom_ppe_model"
    )
    
    print("Training complete.")
    print(f"Best model saved to {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a data.yaml file pointing to your train/val datasets
    DATA_YAML = "data.yaml" 
    
    if os.path.exists(DATA_YAML):
        train_yolo(DATA_YAML)
    else:
        print("Please create a 'data.yaml' file describing your dataset location.")
        print("Example content:")
        print("path: ../datasets/coco128  # dataset root dir")
        print("train: images/train  # train images (relative to 'path')")
        print("val: images/val  # val images (relative to 'path')")
        print("nc: 80  # number of classes")
        print("names: ['person', 'bicycle', 'car', ...]")
