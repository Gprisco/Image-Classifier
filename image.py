from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256, 256))
    
    #Getting image's width and height to do center crop and get the required 224x224 image
    width, height = im.size
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    im = im.crop((left, top, right, bottom))
    im_arr = np.asarray(im)
    
    #Normalize image
    im_arr = im_arr / 255
    im_arr -= np.array([0.485, 0.456, 0.406])
    im_arr /= np.array([0.229, 0.224, 0.225])
    
    return im_arr.transpose((2, 0, 1))