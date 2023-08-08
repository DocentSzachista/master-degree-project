from scripts.setup import Setup
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader


if __name__ == '__main__':
    setup = Setup()
    setup.create_directories()
    # model = setup.download_model()
    transforms = A.Compose([
        ToTensorV2()
    ])
    preprocess = lambda x: transforms(image=np.array(x))["image"].float()/255.0
    cifar = setup.download_test_data(preprocess)
    for rate in setup.config.augumentations[0].make_iterator():
        datset = setup.modify_dataset(setup.config.augumentations[0], cifar, rate)
        loader = DataLoader(datset, batch_size=16, shuffle=False)
