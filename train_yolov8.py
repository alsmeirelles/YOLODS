from ultralytics import YOLO as yolo
import torch
import argparse


def train(config):
    model = yolo(config.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    lr = 0.001 if not config.ftune else 0.0001
    results = model.train(data=config.data,epochs=config.epochs,batch=config.batch,optimizer="Adam",lr0=lr,imgsz=640,save=True,save_period=5, workers=8)
    sucess = model.export()

if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Train an yolov8n model with given \
        dataset.')
    parser.add_argument('--ftune', action='store_true', dest='ftune', default=False,
        help='Finetune a model on top of a given Dataset')
    parser.add_argument('-ep', dest='epochs', type=int,
        help='Number of training epochs (Default: 100)', default=100,required=False)
    parser.add_argument('-bs', dest='batch', type=int,
        help='Batch size (Default: 32)', default=32,required=False)
    parser.add_argument('-data', dest='data', type=str, default='',
        help='Dataset YAML file location.',required=True)
    parser.add_argument('-model', dest='model', type=str, default='yolov8n.pt',
        help='Yolo model to be used. Default is yolov8n.',required=False)

    config, unparsed = parser.parse_known_args()

    train(config)
