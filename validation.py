import torch

def validate(model, validloader, testloader, criterion, optimizer, device):
    valid_loss = 0
    test_loss = 0
    val_accuracy = 0
    test_accuracy = 0

    with torch.no_grad():
        #Evaluation mode
        model.eval()

        print("\n[*] Validating model, this may take a while...")
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)

            #Go forward
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            valid_loss += loss.item()

            #Get accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            val_accuracy += torch.mean(equals.type(torch.FloatTensor))
        else:
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                #Go forward
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                test_loss += loss.item()

                #Get accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

            print(f"Validation Loss: {valid_loss/len(validloader)}")
            print(f"Validation Accuracy: {(val_accuracy/len(validloader)) * 100}")
            print(f"Test Loss: {test_loss/len(testloader)}")
            print(f"Test Accuracy: {(test_accuracy/len(testloader)) *100}")
    print("\n[*] Validation Completed!")
    