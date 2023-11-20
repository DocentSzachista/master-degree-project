import albumentations as A
import numpy as np
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image


class BaseAugumentation:
    def __init__(self, config: dict) -> None:
        self.name = config.get("name")
        self.start_point = config.get("start_point")
        self.finish_point = config.get("finish_point")
        self.step = config.get("step")

    def make_iterator(self):
        return np.arange(self.start_point, self.finish_point, self.step)


class NoiseAugumentation(BaseAugumentation):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.apply_random = config.get("apply_random_on_each_image", False)


class MixupAugumentation(BaseAugumentation):
    def __init__(self, config: dict) -> None:
        transform = A.Compose([
            ToTensorV2()
         ])
        super().__init__(config)
        self.chosen_image = np.asarray(Image.open(config.get("chosen_image")))
        # self.chosen_image = transform(image=np.array(img))["image"].float()/255.0
