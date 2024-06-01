import torch
import sys
import os

# # Get the parent directory of model_keypoint.py
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add the parent directory to the Python path
sys.path.append(module_path)

from utils import model_keypoint


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='keypoint_classifier/pytorch_keypoint_classifier.pth',
        device='cpu'
    ):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(parent_dir, model_path)
        
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.eval()  # Set model to evaluation mode

    def load_model(self, model_path):

        model = model_keypoint.KeyPointClassifier_model(input_size=21*2, num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=self.device))  # Load entire model
        return model

    def __call__(
        self,
        landmark_list,
    ):
        with torch.no_grad():
            input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(self.device)
            output = self.model(input_tensor)
            result_index = torch.argmax(output).item()

        return result_index


