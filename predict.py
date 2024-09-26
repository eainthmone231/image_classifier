# Imports here
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import copy
import argparse
from torchvision.models import vgg16,resnet18
import json
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        as it was trained
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Load image
    image = Image.open(image_path)

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    image = preprocess(image)
    return image

def load_checkpoint(model_type,model_path,device):
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == 'vgg16':
        model = vgg16(pretrained=True)
    elif model_type =='resnet18':
        model = resnet18(pretrained=True)
    else:
        print ("No model available")

    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model



def predict(image_path, model, device,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    image = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probs, top_classes = probabilities.topk(topk, dim=1)

    # Convert to lists
    top_probs = top_probs.cpu().numpy()[0]
    top_classes = top_classes.cpu().numpy()[0]

    with open('class_to_idx.json',"r") as file:
        class_to_idx = json.load(file)

    # Invert class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert indices to classes
    top_labels = [idx_to_class[class_idx] for class_idx in top_classes]

    return top_probs, top_labels


def main(args):

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    device = torch.device("cuda" if args.gpu else "cpu")
    loaded_model = load_checkpoint(args.model_type,args.model_path,device)
    probs, classes = predict(args.img_path, loaded_model,device, args.topk)
    # Get top class names
    top_class_names = [cat_to_name[str(cls)] for cls in classes]
    print("Top class names")
    print(top_class_names)


if __name__ == "__main__":
    # Argument parsing for user inputs
    parser = argparse.ArgumentParser(description='Predict the type of flowers')
    parser.add_argument('model_path', type=str, help='Directory of the model')
    parser.add_argument('--img_path', type=str, help='Path of the image')
    parser.add_argument('--topk',type=int , default=5, help ="Number of top class")
    parser.add_argument('--category_names', default="cat_to_name.json" ,type=str, help='Path to a JSON file mapping labels to categories')
    parser.add_argument('--model_type', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Model architecture')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    args = parser.parse_args()
    main(args)
