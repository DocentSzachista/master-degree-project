import numpy as np
import pandas as pd
from pca_generate import LABELS_CIFAR_10, run

train = "../dataframes/cifar_10.pickle"
test = "../dataframes/test_cifar10.pickle"
values = [2353, 3136, 5066, 6087]
# for value in values:
#     df = pd.read_pickle(
#         f"../dataframes/noises/cifar_10_noised_samples_id_{value}.pickle"
#     )

#     run(
#         train_input_dataframe_path=train,
#         test_input_dataframe_path=test,
#         plot_title=f"Noise image id:{value}",
#         picked_datapoints=df,
#         labels=LABELS_CIFAR_10,
#         output_path=f"../out/only_noise/id_{value}",
#     )

for value in values:
    df = pd.read_pickle(
        f"../dataframes/mixups/cifar_10_mixup_id_{value}.pickle"
    )

    run(
        train_input_dataframe_path=train,
        test_input_dataframe_path=test,
        plot_title=f"mixup image id:{value}",
        picked_datapoints=df,
        labels=LABELS_CIFAR_10,
        output_path=f"../out/only_mixup/id_{value}",
    )