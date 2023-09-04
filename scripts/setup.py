import json
import pathlib
from datetime import datetime
import pandas as pd
from enum import Enum
import torch
import gdown
import torchvision
from torchvision.datasets import VisionDataset
from torch import Tensor
from .augumentations import noise_creation, mixup
from .models import resnet_cifar_10
from torchvision.utils import save_image
from .workflows.cifar_10 import get_features
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms

class SupportedModels(Enum):
    RESNET = "resnet"

class ColorChannels(Enum):
    R = 0
    G = 1
    B = 2

    @staticmethod
    def count_channels(channels: str):
        try:
            return [ColorChannels[channel].value for channel in channels]
        except KeyError:
            raise ValueError("Supplied incorrect channel name")


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
        self.color_channels = ColorChannels.count_channels(
            json_config.get("chosen_color_chanels", "RGB"))
        chosen_images_path = json_config.get("chosen_images", None)
        self.chosen_images = pd.read_pickle(chosen_images_path)["id"].to_numpy() if (
             chosen_images_path is not None) else None


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
        
        self.mask = noise_creation.generate_mask((3, 32, 32), self.config.color_channels)
        self.shuffled_indexes = noise_creation.create_and_shuffle_indexes((32, 32))
        self.columns = ["id", "original_label", "predicted_label", "noise_rate", "classifier", "features"]

 

    def create_directories(self):
        now = datetime.now()
        self.formatted_time = datetime.strftime(now, "%d-%m-%Y_%H:%M")
        for augumentation in self.config.augumentations:
            path = pathlib.Path(f"{self.config.model.value}/{augumentation.name}/{self.formatted_time}")
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath("dataframes").mkdir(parents=False, exist_ok=True)
            path.joinpath("images").mkdir(parents=False, exist_ok=True)


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
        images = dataset.data
        labels = dataset.targets
        
        if type(options) is NoiseAugumentation:
            for index in range(len(dataset)):
                image, label = images[index], labels[index]
                
                processed_image = noise_creation.apply_noise_to_image(
                    self.shuffled_indexes, image, self.mask, rate=int(noise_rate)
                )
                listing.append(processed_image)
                labels.append(label)
                if self.config.save_preprocessing:
                    self._make_image(processed_image, f"./{self.config.model.value}/{options.name}/{self.formatted_time}/images/image_{index}_{label}_noise_{noise_rate}.png")

        elif type(options) is MixupAugumentation:
            for index in range(len(dataset)):
                image, label = images[index], labels[index]
                
                processed_image = mixup.mixup_criterion(noise_rate, image, options.chosen_image)
                listing.append(processed_image)
                labels.append(label)
                if self.config.save_preprocessing:
                    self._make_image(processed_image, f"./{self.config.model.value}/{options.name}/{self.formatted_time}/images/image_{index}_{label}_noise_{noise_rate}.png")
        
        else: 
            raise TypeError("Provided options are not supported")
        
        return listing, labels
        
    def _make_image(self, image: Tensor, image_name: str) -> None:
        save_image(image, image_name)

    def save_results(self, data: dict, options: BaseAugumentation):
        for key, values in data.items():
            df = pd.DataFrame(values, columns=self.columns)
            df.to_pickle(f"./{self.config.model.value}/{options.name}/{self.formatted_time}/dataframes/id_{key}.pickle")


converter = lambda tensor: tensor.detach().cpu().numpy()

class Worker: 

    @staticmethod
    def test_model_data_loader(model, images: list, labels: list, mask_intensity: int, storage: dict):

        index = 0
        with torch.no_grad():
            for image, label in zip(images, labels):
                image = image.cuda().unsqueeze(0)
                logits = model(image)
                features = get_features(model._modules["1"], image)
                _, predicted = torch.max(logits, 1)
                
                storage[index].append([
                        index,  label,
                        predicted.item(), 
                        mask_intensity,
                        converter(logits),
                        converter(features)
                    ])
                index += 1