import json
import pathlib
from datetime import datetime
from enum import Enum

import gdown
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torch import Tensor
from .my_dataset.noised_dataset import NoisedDataset 
from .augumentations import noise_creation
from .models import resnet_cifar_10
import matplotlib.pyplot as plt  
from torchvision.utils import save_image

class SupportedModels(Enum):
    RESNET = "resnet"

    

class SupportedAugumentations(Enum):
    NOISE="noise"
    MIXUP="mixup"


class SupportedDatasets(Enum):
    CIFAR="cifar_10"


class BaseAugumentation: 
    def __init__(self, config: dict) -> None:
        self.name = config.get("name")
        self.start_point = config.get("start_point")
        self.finish_point = config.get("finish_point")
        self.step = config.get("step")

    def make_iterator(self):
        return range(self.start_point, self.finish_point, self.step)

class NoiseAugumentation(BaseAugumentation):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.apply_random = config.get("apply_random_on_each", False)

class MixupAugumentation(BaseAugumentation):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.images_indexes = config.get("images_indexes")


class Config:
    
    supported_augumentations = {
        SupportedAugumentations.MIXUP: MixupAugumentation,
        SupportedAugumentations.NOISE: NoiseAugumentation 
    }

    def __init__(self, json_config: dict) -> None:
        self.model = SupportedModels(json_config.get("model"))
        self.augumentations = [
           self.supported_augumentations.get(SupportedAugumentations(augumentation["name"]))(augumentation) for augumentation in json_config.get("augumentations")
        ]
        self.dataset = SupportedDatasets(json_config.get("dataset"))
        self.model_filename = json_config.get("model_location")
        self.g_drive_hash = json_config.get("model_g_drive")
        self.save_preprocessing = json_config.get("save_preprocessing", False)


class Setup:

    supported_models = {
        SupportedModels.RESNET: resnet_cifar_10.prepare_resnet
    }
    supported_datasets = {
        SupportedDatasets.CIFAR: torchvision.datasets.CIFAR10
    }

    def __init__(self) -> None:
        with open("./config.json", 'r') as file:
            self.config = Config(json.load(file))
        
        self.mask = noise_creation.generate_mask((3, 32, 32))
        self.shuffled_indexes = noise_creation.create_and_shuffle_indexes((32, 32))
 

    def create_directories(self):
        now = datetime.now()
        self.formatted_time = datetime.strftime(now, "%d-%m-%Y_%H:%M")
        for augumentation in self.config.augumentations:
            path = pathlib.Path(f"{self.config.model.value}/{augumentation.name}/{self.formatted_time}")
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath("dataframes").mkdir(parents=False, exist_ok=True)
            path.joinpath("images").mkdir(parents=False, exist_ok=True)
        # path = pathlib.Path(self.config.model_filename)
        # path.mkdir(exist_ok=True)

    def download_model(self):
        model_function = self.supported_models.get(self.config.model)
        if model_function is None:
            raise KeyError("Provided dataset is not supported")
        if self.config.g_drive_hash is not None:
            filename = gdown.download(id=self.config.g_drive_hash)
            return model_function(f"./{filename}")
        else:
            return model_function(self.config.model_filename)
    
    def download_test_data(self, preprocess):
        data_function = self.supported_datasets.get(self.config.dataset)
        if data_function is None:
            raise KeyError("Provided dataset is not supported")
        return data_function("./datasets", train=False, download=True, transform=preprocess) 

    def modify_dataset(self, options: BaseAugumentation, 
                           dataset: VisionDataset, noise_rate: float
                           ):    
        """Transforms images according to passed options, yields dataset with """
        
        listing = []
        labels = []
        
        if type(options) is NoiseAugumentation:
            for index in range(len(dataset)):
                image, label = dataset[index]
                processed_image = noise_creation.apply_noise_to_image(
                    self.shuffled_indexes, image, self.mask, rate=int(noise_rate)
                )
                listing.append(processed_image)
                labels.append(label)
                if self.config.save_preprocessing:
                    self._make_image(processed_image, f"./{self.config.model.value}/{options.name}/{self.formatted_time}/images/image_{index}_{label}_noise_{noise_rate}.png")

        elif type(options) is MixupAugumentation:
            raise NotImplementedError("Not implemented yet")
        
        else: 
            raise TypeError("Provided options are not supported")
        
        return NoisedDataset(listing, labels)
        
    def _make_image(self, image: Tensor, image_name: str) -> None:
        save_image(image, image_name)