import numpy as np
from PIL import Image
import torchvision.transforms as transforms


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
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        super().__init__(config)
        img = Image.open(config.get("chosen_image"))
        self.chosen_image = transform(img)
