from scripts.setup import Setup, Worker
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

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

    for augumentation in setup.config.augumentations:
        storage = {k: [] for k in range(len(cifar))}
        for rate in augumentation.make_iterator():
            images, labels = setup.modify_dataset(augumentation, cifar, rate)
            Worker.test_model_data_loader(model, images, labels, rate, storage)
        setup.save_results(storage, augumentation)

