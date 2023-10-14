from scripts.setup import Setup, Worker
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from scripts.plots.barplot import run

if __name__ == '__main__':

    setup = Setup()
    setup.create_directories()
    model = setup.download_model()
    transforms = A.Compose([
        ToTensorV2()
    ])
    preprocess = lambda x: transforms(image=np.array(x))["image"].float()/255.0
    cifar = setup.download_test_data(preprocess)
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
        images = [cifar[i][0] for i in range(len(cifar))]
        cifar.data = images # Done to make sure that images are as tensors, not numpy arrays
    for augumentation in setup.config.augumentations:
        for rate in augumentation.make_iterator():
            images, labels = setup.modify_dataset(augumentation, cifar, rate, indexes=setup.config.chosen_images)
            Worker.test_model_data_loader(model, images, labels, rate, storage, indexes=indexes)
        setup.save_results(storage, augumentation)
        storage = {k: [] for k in storage.keys()}

    run()
