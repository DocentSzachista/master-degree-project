import random

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision
import workflows.utils as utils
from albumentations.pytorch import ToTensorV2
from augumentations.mixup import mixup_criterion
from augumentations.noise_creation import (apply_noise_to_image,
                                           create_and_shuffle_indexes,
                                           generate_mask)
from models import resnet_cifar_10

columns = [
    "id",
    "original_label",
    "predicted_label",
    "noise_rate",
    "classifier",
    "features",
]
old_columns = ["id", "original_label", "classifier", "features"]
converter = lambda tensor: tensor.detach().cpu().numpy()


def save_to_pickle_file(data: list, columns: list, filepath: str):
    """Save collected data about model performance to pickle file"""
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(filepath)
    return df


def get_features(model, data):
    """Retrieves feature values from avgpool layer of network."""
    for name, module in model._modules.items():
        data = module(data)
        # print(name, data.shape)
        if name == "avgpool":
            # tu są cechy które chcę wyciągnąć z sieci
            data = data.view(-1, 2048)
            return data


def test_single(model, image):
    """Test single augumented image.

    Returns
    -------
    logits for given picture and features from avgpooling layer.
    """
    with torch.no_grad():
        image = image.cuda().unsqueeze(0)
        logits = model(image)
        features = get_features(
            model._modules["1"], image
        )  # TODO PARAMETRISE THIS LINE
        _, predicted = torch.max(logits, 1)
        # print(logits.shape)
        # print(features.shape)
        return logits, features, predicted


def retrieve_images_from_dataloader(preprocess, image_ids: list):
    test_set = torchvision.datasets.CIFAR10(
        root="./", train=False, download=True, transform=ToTensorV2()
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    test_dataset = test_loader.dataset
    images = [test_dataset[i][0] for i in image_ids]
    labels = [test_dataset[i][1] for i in image_ids]

    return labels, images


def single_run_noise(
    model,
    images,
    mask,
    shuffled_indexes: list,
    labels: list,
    image_ids: list,
    to_save: dict,
    step: int,
):
    #   fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for index in range(len(images)):
        image = apply_noise_to_image(shuffled_indexes, images[index], mask, step)
        # axs[index].imshow(image.permute(1, 2, 0))
        logits, features, predicted = test_single(model, image)
        to_save[image_ids[index]].append(
            [
                image_ids[index],
                labels[index],
                predicted.item(),
                int(step / 1024 * 100),
                converter(logits),
                converter(features),
            ]
        )
        print(f"Predicted: {predicted}. Original {labels[index]}")
    # plt.show()


def retrieve_chosen_images_ids(filepath: str):
    chosen = pd.read_pickle(filepath)
    return chosen["id"].to_numpy()


## TODO: ADD functions for data augumentation preparation and
def start_workflow_noise(filepath: str):
    """Makes experiments with using mask noise size of ."""
    utils.set_workstation("cuda:0")
    model = resnet_cifar_10.prepare_resnet()
    chosen_images = retrieve_chosen_images_ids(filepath)
    mask = generate_mask((3, 32, 32))
    shuffled_indexes = create_and_shuffle_indexes(mask.shape[1:])
    data_transforms = A.Compose([ToTensorV2()])
    preprocess = lambda x: data_transforms(image=np.array(x))["image"].float() / 255.0
    images, labels = retrieve_images_from_dataloader(preprocess, chosen_images)

    values_range = list(range(50, 500, 50))
    to_save = {k: [] for k in chosen_images}
    for value in values_range:
        single_run_noise(
            model, images, mask, shuffled_indexes, labels, chosen_images, to_save, value
        )

    for key, noised_data in to_save.items():
        chosen = chosen[old_columns]
        retrieved = chosen.loc[chosen.id == key].values.flatten().tolist()
        retrieved.insert(2, retrieved[1])
        retrieved.insert(3, 0)
        whole_data = [retrieved] + noised_data
        df = save_to_pickle_file(
            whole_data, columns, f"cifar_10_noised_samples_id_{key}"
        )


def start_workflow_mixup(filepath: str, filepath_2: str):
    utils.set_workstation("cuda:0")
    model = resnet_cifar_10.prepare_resnet()
    chosen_images = retrieve_chosen_images_ids(filepath)
    chosen_mixed_images = retrieve_chosen_images_ids(filepath_2)
    data_transforms = A.Compose(
        [
            # A.GaussNoise(always_apply=True, per_channel=True, var_limit=(0, 1000)),
            ToTensorV2()
        ]
    )
    preprocess = lambda x: data_transforms(image=np.array(x))["image"].float() / 255.0
    images, labels = retrieve_images_from_dataloader(preprocess, chosen_images)
    mixed_images, mixed_labels = retrieve_images_from_dataloader(
        preprocess, chosen_mixed_images
    )

    to_save = {k: [] for k in chosen_images}

    for lambda_ in range(0, 1, 0.1):
        for index in range(len(images)):
            image = mixup_criterion(lambda_, images[index], mixed_images[0])
            # axs[index].imshow(image.permute(1, 2, 0))
            logits, features, predicted = test_single(model, image)
            to_save[chosen_images[index]].append(
                [
                    chosen_images[index],
                    labels[index],
                    predicted.item(),
                    int(lambda_ * 100),
                    converter(logits),
                    converter(features),
                ]
            )
            print(f"Predicted: {predicted}. Original {labels[index]}")

    for key, noised_data in to_save.items():
        chosen_images = chosen_images[old_columns]
        retrieved = chosen_images.loc[chosen_images.id == key].values.flatten().tolist()
        retrieved.insert(2, retrieved[1])
        retrieved.insert(3, 0)
        whole_data = [retrieved] + noised_data
        df = save_to_pickle_file(
            whole_data, columns, f"cifar_10_mixed_samples_id_{key}"
        )
