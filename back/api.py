import torch

import numpy as np

import albumentations as alb

from src.bicanet import BiCADenseNet


class Segmentator:
    PORTRAIT_MEAN, PORTRAIT_STD = (0.5107, 0.4506, 0.4192), (0.3020, 0.2839, 0.2802)

    _image_transform = alb.Compose([
        alb.Normalize(mean=PORTRAIT_MEAN, std=PORTRAIT_STD, always_apply=True),
        alb.Resize(256, 192, always_apply=True)
    ])

    def __init__(self, path, dtype=torch.float32, device=torch.device('cpu')):
        self.path = path
        self.dtype = dtype
        self.device = device

        model_checkpoint = torch.load(path, map_location=torch.device('cpu'))

        self.model = BiCADenseNet(num_classes=2, mcfb_h=256, mcfb_w=192)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])

        self.model.to(device=device, dtype=dtype)
        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            image = Segmentator._image_transform(
                image=np.array(image),
            )['image']

            image_tensor = torch.tensor(image, dtype=self.dtype, device=self.device).permute(2, 0, 1)
            segmentation = torch.softmax(self.model(image_tensor[None, :, :, :]), dim=1)[0, 1]

            return segmentation.cpu().numpy()
