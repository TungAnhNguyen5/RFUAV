# A script to do the inference on the spectrogram image or binary raw frequency data pack using trained classify model or two-stage detector model.

from utils.benchmark import Classify_Model
from utils.TwoStagesDetector import TwoStagesDetector

def main():

     # doing a inference on spectrogram image or binary raw frequency data pack using the trained classify model
    source = './dataset/valid/YunZhuo-H16'  # for inference test
    test = Classify_Model(cfg='./configs/exp1.2_ResNet34.yaml',
                          weight_path='./models/exp1.2_ResNet34/best_model.pth')
    test.inference(source=source, save_path='./res/')  # for inference test

    # # doing a two-stage detector inference on the binary raw frequency data pack using the trained detector and classify model
    # cfg_path = '../example/two_stage/sample.json'
    # TwoStagesDetector(cfg=cfg_path)


if __name__ == '__main__':
    main()