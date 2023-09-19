from typing import List, Tuple
import numpy as np
import pandas as pd

def get_missing_values_indexes (data: pd.DataFrame) -> List[int]:

    missing: List[int] = []

    for index, row in data.iterrows():

        # check each value
        for column in data:
            ele = row[column]

            try:
                float(ele)

            # not a number
            except ValueError:
                missing.append(index)

    return missing



