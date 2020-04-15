import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from validation import validate

def get_model(arch):
    '''
        Gets the architecture string given by the user
        
        Returns the pretrained model from torchvision.models
    '''
    switcher = {
        "vgg16": models.vgg16,
        "alexnet": models.alexnet,
        "resnet": models.resnet18
    }
    
    model = switcher.get(arch, False)
    
    if not model:
        print("We don't support this architecture yet, try with:")
        
        for key in switcher:
            print(key)
        
        return
    
    return model(pretrained=True)

def prepare_network(model, hidden_units):
    '''
        Gets the model and the number of hidden units
        
        Returns the model with a personalized classifier
    '''
    print("[*] Preparing the network...")
    
    #Freeze features parameters
    for param in model.parameters():
        param.requires_grad = False

    fc = nn.Sequential(nn.Linear(25088, hidden_units),
                       nn.ReLU(),
                       nn.Dropout(p=0.2),
                       nn.Linear(hidden_units, hidden_units),
                       nn.ReLU(),
                       nn.Dropout(p=0.2),
                       nn.Linear(hidden_units, 102),
                       nn.LogSoftmax(dim=1))

    model.classifier = fc
    print("Finished.")

def train(model, data_dir, epochs, gpu, lr):
    trainloader, validloader, testloader = load_datasets(data_dir)
    
    if not trainloader:
        return
    
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

    print(f"\n[*] Training on device: {device}... this will be a long operation based on the epochs you specified.")
    
    #Define criterion for calculating loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device)
    
    with active_session():
        for e in range(epochs):
            running_loss = 0

            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"\nEpoch {e+1}/{epochs}")
            print(f"Train Loss: {running_loss/len(trainloader)}")
    print("\n[*] Finished Training")
    
    validate(model, validloader, testloader, criterion, optimizer, device)
    
def load_datasets(data_dir):
    # Get different directories for all the steps
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Specify data transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                           ])

    test_val_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])
    
    try:
        # Load the datasets with ImageFolder
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(valid_dir, transform=test_val_transforms)
        test_dataset = datasets.ImageFolder(test_dir, transform=test_val_transforms)
    except FileNotFoundError:
        print("[!] Seems like the directory does not contain required folders")
        print("\tTry structuring it like this:")
        print("-root/")
        print("----train/")
        print("----valid/")
        print("----test/")
        return None, None, None

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=35, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=26, shuffle=True)
    
    return trainloader, validloader, testloader