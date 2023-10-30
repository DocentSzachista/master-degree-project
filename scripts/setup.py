import json
import pathlib
from datetime import datetime

import gdown
import pandas as pd
import torch
import torchvision
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from .augumentations import mixup, noise_creation
from .models import resnet_cifar_10
from .workflows.augumentations import (BaseAugumentation, MixupAugumentation,
                                       NoiseAugumentation)
from .workflows.cifar_10 import get_features
from .workflows.enums import (ColorChannels, SupportedAugumentations,
                              SupportedDatasets, SupportedModels)
from .workflows.utils import set_workstation


class Config:

    supported_augumentations = {
        SupportedAugumentations.MIXUP: MixupAugumentation,
        SupportedAugumentations.NOISE: NoiseAugumentation
    }

    def __init__(self, json_config: dict) -> None:
        self.model = SupportedModels(json_config.get("model"))
        self.tag = json_config.get("tag", "base")
        self.augumentations = [
            self.supported_augumentations.get(SupportedAugumentations(augumentation["name"]))(augumentation)
            for augumentation in json_config.get("augumentations")]
        self.dataset = SupportedDatasets(json_config.get("dataset"))
        self.model_filename = json_config.get("model_location")
        self.g_drive_hash = json_config.get("model_g_drive")
        self.save_preprocessing = json_config.get("save_preprocessing", False)
        self.color_channels = ColorChannels.count_channels(
            json_config.get("chosen_color_chanels", "RGB"))
        chosen_images_path = json_config.get("chosen_images", None)
        self.chosen_images = pd.read_pickle(chosen_images_path)["id"].to_numpy() if (
            chosen_images_path is not None) else None
        self.image_dim = json_config.get("image_dim", [3, 32, 32])
        self.training_df = pd.read_pickle(json_config.get("train_file"))


class Setup:

    supported_models = {
        SupportedModels.RESNET: resnet_cifar_10.prepare_resnet
    }
    supported_datasets = {
        SupportedDatasets.CIFAR: torchvision.datasets.CIFAR10
    }

    def __init__(self, config_file="./config.json") -> None:
        with open(config_file, 'r') as file:
            self.config = Config(json.load(file))

        self.mask = noise_creation.generate_mask(
            self.config.image_dim, self.config.color_channels)
        self.shuffled_indexes = noise_creation.create_and_shuffle_indexes(
            (32, 32))
        self.columns = ["id", "original_label", "predicted_label",
                        "noise_rate", "classifier", "features", "noise_percent"]

    def create_directories(self):
        now = datetime.now()
        self.formatted_time = datetime.strftime(now, "%d-%m-%Y_%H:%M")
        for augumentation in self.config.augumentations:
            path = pathlib.Path(
                f"{self.config.model.value}-{self.config.tag}/{augumentation.name}")
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
                       dataset: VisionDataset, noise_rate: float, indexes: list
                       ):
        """Transforms images according to passed options, yields dataset with """

        listing = []
        images = dataset.data
        labels = dataset.targets
        if type(options) is NoiseAugumentation:
            for index in range(len(dataset)):
                image, label = images[index], labels[index]
                processed_image = noise_creation.apply_noise_to_image(
                    self.shuffled_indexes, image, self.mask, rate=int(
                        noise_rate)
                )
                listing.append(processed_image)
                # labels.append(label)
                if self.config.save_preprocessing:
                    ids = indexes[index] if indexes is not None else index
                    self._make_image(
                        processed_image,
                        f"./{self.config.model.value}-{self.config.tag}/{options.name}/images/image_{ids}_{label}_noise_{round(noise_rate, 2)}.png")

        elif type(options) is MixupAugumentation:
            for index in range(len(dataset)):
                image, label = images[index], labels[index]
                processed_image = mixup.mixup_criterion(
                    noise_rate, options.chosen_image, image)
                listing.append(processed_image)
                if self.config.save_preprocessing:
                    ids = indexes[index] if indexes is not None else index
                    self._make_image(
                        processed_image,
                        f"./{self.config.model.value}-{self.config.tag}/{options.name}/images/image_{ids}_{label}_noise_{noise_rate}.png")

        else:
            raise TypeError("Provided options are not supported")

        return listing, labels

    def _make_image(self, image: Tensor, image_name: str) -> None:
        save_image(image, image_name)

    def save_results(self, data: dict, options: BaseAugumentation):
        for key, values in data.items():
            df = pd.DataFrame(values, columns=self.columns)
            if isinstance(options, NoiseAugumentation):
                df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / 1024, 2))
            elif isinstance(options, NoiseAugumentation):
                df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / 100, 2))

            df.to_pickle(
                f"./{self.config.model.value}-{self.config.tag}/{options.name}/dataframes/id_{key}.pickle")


def converter(tensor): return tensor.detach().cpu().numpy()


class Worker:

    @staticmethod
    def test_model_data_loader(model, images: list, labels: list, mask_intensity: int, storage: dict, indexes: list):
        set_workstation("cuda:0")
        with torch.no_grad():
            for image, label, id in zip(images, labels, indexes):
                image = image.cuda().unsqueeze(0)
                logits = model(image)
                features = get_features(model._modules["1"], image)
                _, predicted = torch.max(logits, 1)

                storage[id].append([
                    id,  label,
                    predicted.item(),
                    mask_intensity,
                    converter(logits),
                    converter(features),
                    100*mask_intensity
                ])

    @staticmethod
    def test_model_with_data_loader(model, data_loader: DataLoader, mask_intensity: int, storage: dict):
        set_workstation("cuda:0")
        model.eval()
        model.to("cuda:0")
        for idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
                logits = model(inputs)
                features = get_features(model._modules['1'], inputs)
                _, predicted = torch.max(logits, 1)
                storage[idx].append([
                    f"{idx}_{round(mask_intensity, 2)}", targets,
                    predicted.item(),
                    mask_intensity,
                    converter(logits),
                    converter(features),
                    100*mask_intensity
                ])
