import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def separate_data_by_tags (
        data: pd.DataFrame,
        tags: List[str],
        class_column: str
    ) -> Dict[str, pd.DataFrame]:

    data_groups: Dict[str, pd.DataFrame] = {}
    group: pd.DataFrame

    # iterate tags

    for tag in tags:

        group = pd.DataFrame()

        for index, row in data.iterrows():

            # add row to group if tag matches
            if row[class_column] == tag:
                group = pd.concat([group, pd.DataFrame(row).T])

        data_groups[tag] = group
        
    return data_groups


def balance_data_by_dropping_rows (
        data: pd.DataFrame,
        tags: List[str],
        data_groups: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:

    # get min and max
    
    min = float("inf")
    max = 0

    for tag in tags:
        size = len(data_groups[tag].index)
        if size > max:
            max = size
        if size < min:
            min = size

    # drop

    for tag in tags:
        size_diff = len(data_groups[tag].index) - min
        data_groups[tag] = data_groups[tag].iloc[size_diff:]


def drop_rows_less_50(
        data: pd.DataFrame,
        tags: List[str],
        data_groups: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
    
    classes = []
    c = 0
    
    for tag in tags:
        c = c + 1
        if len(data_groups[tag].index) < 50:
            classes.append(c)
            data_groups.pop(tag)
            
    for c in classes:
        tags.remove(c)


def get_training_and_testing_groups (
        percentage_train: float,
        tags: List[str],
        data_groups: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_train = pd.DataFrame()
    data_test  = pd.DataFrame()

    data_train_size = len(data_groups[tags[0]].index) * percentage_train

    for tag in tags:
        sub_train, sub_test = np.split (data_groups[tag], [int(data_train_size)])

        data_train = pd.concat([data_train, sub_train])
        data_test = pd.concat([data_test, sub_test])

    return data_train, data_test
