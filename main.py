from scripts.setup import Setup, Worker
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from scripts.plots.barplot import run
import copy
from torch.utils.data import DataLoader
import torch
from scripts.augumentations.noise_creation import apply_noise_to_image
from scripts.augumentations.mixup import mixup_criterion

if __name__ == '__main__':

    setup = Setup()
    torch.cuda.empty_cache()
    setup.create_directories()
    model = setup.download_model()
    transforms = A.Compose([
        ToTensorV2()
    ])
    def preprocess(x): return transforms(image=np.array(x))["image"].float()/255.0
    cifar = setup.download_test_data(preprocess)
    print(cifar.data.shape)
    # part where chosen images are picked.
    if setup.config.chosen_images is not None:
        images = [cifar[i][0] for i in setup.config.chosen_images]
        labels = [cifar[i][1] for i in setup.config.chosen_images]
        cifar.data = images
        cifar.targets = labels
        indexes = setup.config.chosen_images
        storage = {k: [] for k in indexes}
    else:
        indexes = list(range(len(cifar)))
        storage = {k: [] for k in indexes}
        # images = np.vstack([cifar[i][0] for i in range(len(cifar))])
        # cifar.data = images  # Done to make sure that images are as tensors, not numpy arrays
        copy_cifar = copy.deepcopy(cifar)
    for augumentation in setup.config.augumentations:
        iterator = augumentation.make_iterator()
        for image_id in indexes:
            augumented_class = []
            converted_ids = []
            starting_image = cifar.data[image_id]
                # processed_image = apply_noise_to_image(
                #     setup.shuffled_indexes, starting_image, setup.mask.numpy(), rate)
            for rate in iterator:
                processed_image= mixup_criterion( rate, augumentation.chosen_image, starting_image, 
                        )
                augumented_class.append(processed_image)
                converted_ids.append(f"{image_id}_{rate}")
                if setup.config.save_preprocessing:
                    setup._make_image(
                        processed_image,
                        f"./{setup.config.model.value}-{setup.config.tag}/{augumentation.name}/images/image_{image_id}_noise_{round(rate, 2)}.png")
            # break
            labels = cifar.targets #[cifar.targets[image_id] for i in range(0, len(iterator))]
            stack = np.array(augumented_class)
    #         # images, labels = setup.modify_dataset(augumentation, cifar, rate, indexes=setup.config.chosen_images)
    #         # stack = np.vstack(images)
            copy_cifar.data = stack
            copy_cifar.targets = labels
            data_loader = DataLoader(
                copy_cifar, batch_size=32, shuffle=False, drop_last=True
            )
    #         # Worker.test_model_data_loader(model, images, labels, rate, storage, indexes=indexes)
            to_save = Worker.test_model_with_data_loader(
                model=model, data_loader=data_loader, 
                mask_intensity=iterator, converted_ids=converted_ids
            )
            setup.save_results_gpu(to_save, augumentation, image_id, labels[image_id])


        # setup.save_results(storage, augumentation)
        # storage = {k: [] for k in storage.keys()}

    # run()
