import argparse

class TrainParser:
    def __init__(self):
        description = "Flower Classifier Trainer CLI. Train a classifier for over 100 flower types."
        usage = "\nBasic usage: `python train.py data_directory`"
        
        parser = argparse.ArgumentParser(description=description+usage)

        #Adding arguments
        parser.add_argument("data_directory", type=str, help="The path of the dataset which contains train/ valid/ and test/ directories.")
        parser.add_argument("--arch", type=str, help="Choose the network architecture.", default="vgg16")
        parser.add_argument("--save-dir", type=str, help="Choose a directory in which to save the checkpoint.", default="./")
        parser.add_argument("--hidden-units", type=int, help="Number of units for the hidden layers.", default=512)
        parser.add_argument("--gpu", type=bool, help="Specify if you want to train on GPU or not.", default=False, const=True, nargs="?")
        parser.add_argument("--epochs", type=int, help="The number of iterations for the training part.", default=3)
        parser.add_argument("--learning-rate", type=float, help="The learning rate for the optimizer.", default=0.003)
        
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

class PredictParser:
    def __init__(self):
        description = "Flower Classifier CLI. Predict over 100 flower types."
        usage = "Basic usage: `python predict.py /path/to/image checkpoint.pth`"
        
        parser = argparse.ArgumentParser(description=description+usage)
        
        parser.add_argument("path_to_image", type=str, help="The path to the image to use as an input.")
        parser.add_argument("checkpoint", type=str, help="The path to the model checkpoint to use to classify the input image.")
        parser.add_argument("--top_k", type=int, help="Display the top_k results.", default=1)
        parser.add_argument("--gpu", type=bool, help="Use GPU for inference.", default=False, const=True, nargs="?")
        parser.add_argument("--category_names", type=str, help="Use a mapping of categories to real name.", default="cat_to_name.json")
        
        self.parser = parser
        
    def parse(self):
        return self.parser.parse_args()
