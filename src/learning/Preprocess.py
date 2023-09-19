import pandas as pd
from typing import List, Dict

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
