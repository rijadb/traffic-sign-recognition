import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from class_names import CLASS_NAMES 

from model import TrafficSignCNN


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth"
)


model = TrafficSignCNN()                            # create model


model.load_state_dict(torch.load(MODEL_PATH))       # load learned weights back to CNN


model.eval()                                        # switch to prediction/evaluation mode

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image)

    image = image.unsqueeze(0)                      # adds a new dimension (batch)

    return image

def predict(image_path):

    image = preprocess_image(image_path)

    with torch.no_grad():                           # so pythorch wont save gradients

        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

        """
        print("\nTop 5 probabilities:")

        top_probs, top_classes = torch.topk(probabilities, 5)

        for i in range(5):
            print(
                f"Class {top_classes[0][i].item()} : "
                f"{top_probs[0][i].item()*100:.2f}%"

        -used to show 5 classes with biggest prediction %        
        """
    

    return predicted.item(), confidence.item()

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predicted_class, confidence = predict(image_path)

    print(f"\nPrediction: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")