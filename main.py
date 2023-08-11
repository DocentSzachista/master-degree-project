from scripts.setup import Setup, Worker
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader


if __name__ == '__main__':
    
    setup = Setup()
    setup.create_directories()
    model = setup.download_model()
    transforms = A.Compose([
        ToTensorV2()
    ])
    preprocess = lambda x: transforms(image=np.array(x))["image"].float()/255.0
    cifar = setup.download_test_data(preprocess)
    
    storage = {k: [] for k in range(len(cifar))}

    for rate in setup.config.augumentations[0].make_iterator():
        images, labels = setup.modify_dataset(setup.config.augumentations[0], cifar, rate)
        Worker.test_model_data_loader(model, images, labels, rate, storage)
        break
    setup.save_results(storage, setup.config.augumentations[0])


