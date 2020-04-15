import torch
import json
import numpy as np
from image import process_image
from parser import PredictParser
from checkpoint_handler import load_model

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = torch.Tensor(process_image(image=image_path)).unsqueeze_(0)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    model.to(device)
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(image)
    
    ps = np.exp(logps)
    
    probs, classes = ps.topk(topk, dim=1)
    
    return np.asarray(probs[0])*100, np.asarray(classes[0])

def main():
    parser = PredictParser()
    args = parser.parse()
    
    # Load model and category to names json file
    try:
        model = load_model(args.checkpoint)
        
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    except FileNotFoundError:
        print("[!] Missing image or category_names file")
        return

    # Start the prediction
    probs, classes = predict(args.path_to_image, model, args.top_k, args.gpu)
    
    labels = list()

    for c in classes:
        labels.append(cat_to_name[str(c+1)])
    
    print("Output:")
    
    for i in range(len(labels)):
        print(f"\n{labels[i]}\t{probs[i]}")
    
if __name__ == "__main__":
    main()
