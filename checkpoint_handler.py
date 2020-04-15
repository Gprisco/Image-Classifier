import torch
from training_functions import get_model

def save_model(model, save_dir, arch):
    path = save_dir + "checkpoint.pth"
    
    # Save the checkpoint
    checkpoint = {'classifier': model.classifier,
                  'arch': arch,
                  'state_dict': model.classifier.state_dict()}

    try:
        torch.save(checkpoint, path)
        print(f"[*] Model successfully saved at {path}")
    except FileNotFoundError:
        print("[!] The directory does not exist, try creating it.")
    
def load_model(path):
    try:
        checkpoint = torch.load(path)
    except FileNotFoundError:
        print(f"[!] No checkpoint at {path}")
        return
    
    model = get_model(checkpoint['arch'])
    
    if not model:
        return
    
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    return model