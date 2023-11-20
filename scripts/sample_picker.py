import numpy as np
import pandas as pd

def pick_chosen_label(dataframe: pd.DataFrame, label: int) -> pd.DataFrame:
    """ Pick chosen label sorted descending.   

        Function filters dataframe to retrieve only desired label and
        later sorts it by its confidence in classifier column

        Parameters
        ----------
        label: int
            Class label to retrieve from dataframe.

        Returns
        -------
        Dataframe with selected label and sorted from highest confidence
        to the lowest.

    """
    dataframe = dataframe[dataframe["original_label"] == label]  
    sorted_indices = np.argsort(dataframe['classifier'].apply(lambda x: x[label]))[::-1]
    return dataframe.iloc[sorted_indices]

