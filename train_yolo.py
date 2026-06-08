from utils.trainer import DetTrainer


def main():
    trainer = DetTrainer(
        model_name="yolo",
        dataset_dir="./rfuav_yolo"
    )

    trainer.train(save_dir="./models/yolo_exp")


if __name__ == "__main__":
    main()