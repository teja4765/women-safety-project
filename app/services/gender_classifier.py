"""
Gender classification service using lightweight CNN
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Optional
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class GenderClassifier(nn.Module):
    """Lightweight CNN for gender classification"""
    
    def __init__(self, num_classes=2):
        super(GenderClassifier, self).__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class GenderClassifierService:
    """Gender classification service"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    async def initialize(self):
        """Initialize the gender classification model"""
        try:
            # Create model
            self.model = GenderClassifier(num_classes=2)
            self.model.to(self.device)
            self.model.eval()
            
            # Load pretrained weights if available
            try:
                checkpoint = torch.load('models/gender_classifier.pth', map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded pretrained gender classifier")
            except FileNotFoundError:
                logger.warning("No pretrained gender classifier found, using random weights")
                # Initialize with random weights for now
                # In production, this should be trained on appropriate data
            
            logger.info("Gender classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize gender classifier: {e}")
            raise
    
    async def classify(self, person_crop: np.ndarray) -> float:
        """
        Classify gender of a person crop
        
        Args:
            person_crop: BGR image crop of person
            
        Returns:
            Probability of being female (0.0 to 1.0)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Preprocess image
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Return probability of being female (class 1)
                female_prob = probabilities[0][1].item()
                
                # Apply confidence threshold and smoothing
                # If confidence is too low, return neutral (0.5)
                max_prob = torch.max(probabilities[0]).item()
                if max_prob < 0.6:  # Low confidence threshold
                    return 0.5
                
                return female_prob
                
        except Exception as e:
            logger.error(f"Error in gender classification: {e}")
            return 0.5  # Return neutral on error
    
    def preprocess_for_training(self, image_path: str, label: int) -> tuple:
        """
        Preprocess image for training
        
        Args:
            image_path: Path to image file
            label: 0 for male, 1 for female
            
        Returns:
            (tensor, label) tuple
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image)
            return tensor, label
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None, None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": "GenderClassifier",
            "input_size": "64x64",
            "num_classes": 2,
            "device": str(self.device),
            "classes": ["male", "female"]
        }
