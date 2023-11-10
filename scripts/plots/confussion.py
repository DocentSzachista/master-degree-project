from ..utils.constants import LABELS_CIFAR_10
from ..setup import Config
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_confussion_matrix(y_true: list, y_predicted: list, labels: dict,  filename: str):
    """Generate confussion_matrix heatmap.

        :param y_predicted: predicted classes by neural network
        :type y_predicted: list 1-D of integers in range (0, 9)
        :param y_predicted: labeled classes in test set
        :type y_predicted: list 1-D of integers in range (0, 9)
    """
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=list(labels.keys()))
    print(cf_matrix)
    df_cm = pd.DataFrame(
        cf_matrix, #/ np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in labels.values()],
        columns=[i for i in labels.values()]
    )
    plt.figure(figsize=(40, 16))
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = list(labels.values()))    
    cm_display.plot()
# sn.heatmap(
#         df_cm, annot=True
#     )
    print("save")
    plt.savefig(f"{filename}.png")



def run(conf: Config, condition: dict ):

    for augumentation in conf.augumentations:
        path = f"./{conf.model.name.lower()}-{conf.tag}/{augumentation.name}"
        print(path)
        files = []
        for image_class in LABELS_CIFAR_10.keys():
            files.extend([os.path.join(f"{path}/dataframes/{image_class}", file)
                     for file in os.listdir(f"{path}/dataframes/{image_class}")   ])
        dfs = []
        # for iteration conf.augumentations[0].make_iterator()
        for file in files:
            new_df = pd.read_pickle(file)
            
            # print(len(new_df.index))
            dfs.append(new_df)
        df = pd.concat(dfs, ignore_index=True)
        for k, v in condition.items():
            subset = df.loc[df['noise_rate'] < v]
            y_true = subset['predicted_label'].values
            y_pred = subset['original_label'].values
            generate_confussion_matrix(y_true, y_pred, LABELS_CIFAR_10, f"./test-{k}-numbers")