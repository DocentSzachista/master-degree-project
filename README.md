# Master degree thesis


## How to launch it?

1. Create virtual environment

```
    python3 -m venv venv
```

2. Activate it

```
    source venv/bin/activate 
```

3. Install requirements.txt
```
    pip3 install -r requirments.txt
```

4. Launch script

```
    python3 main.py 
```

## Config.json file

### Required options:

  -  "model": name of the model to be processed.

  -  "model_g_drive": Id of file on Google drive that contains model to be downloaded,

  -  "augumentations": list of image augmentations to be applied. Currently supported are `noise` and `mixup` augumentations

    Required fields for augumentation:
        - name: name of the augumentation
        - start_point: starting amount of pixels affected by augumentation
        - finish_point: end amount of pixels affected by augumentation
        - step: how many pixels affected increases
  -  "dataset": dataset name that we want to work on. For now only supported is `"CIFAR10"`
  -  "image_dim": Image dimensions
    Argument specific for `mixup` augumentation
    - "chosen_image": path to image to blend in every image in dataset.

  ### Optional fields:
  - "apply_random_on_each_image": for each image generate a new mask
  - "chosen_color_chanels": color channels that should be noised, default value: `"RGB"` 
  - "save_preprocessing": Save preprocessed images. Defaults to `false` 
