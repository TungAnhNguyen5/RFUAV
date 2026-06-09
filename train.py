# A script to train the detect model and classification model on spectrogram image.
import os

from utils.trainer import CustomTrainer
from utils.trainer import DetTrainer


def main():

    # make the model and dataset directory if not exist
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./dataset', exist_ok=True)

    # classification Trainer
    # Make the file path of the model name
    model = CustomTrainer(cfg='./configs/exp3.1_ResNet18.yaml')
    model.train()

    # # Detection Trainer
    # save_dir = ''
    # model = DetTrainer(model_name='yolo', dataset_dir = '')
    # model.train(save_dir=save_dir)


if __name__ == '__main__':
    main()

