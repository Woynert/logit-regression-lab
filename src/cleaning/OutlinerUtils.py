from typing import List
import numpy as np
import pandas as pd

SENS = 1.5

def is_outliner (value, q1, q3, iqr) -> bool:

    return (value < (q1 - SENS * iqr)) or (value > (q3 + SENS * iqr))

def get_outliners (data: List[float]) -> List[int]:
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    return [idx for idx, v in enumerate(data) if is_outliner(v, q1, q3, iqr)]

def get_outliners_indexes (data: pd.DataFrame) -> List[int]:

    outliners: List[int] = []

    # get indexes for all outliners

    for entry in data:
        if str(entry) == "C33":
            continue
        new_outliners = get_outliners(data[str(entry)].tolist())
        outliners.extend(new_outliners)

    outliners = list(set(outliners))
    return outliners

def remove_outliners (data: pd.DataFrame) -> pd.DataFrame:

    outliners: List[int] = get_outliners_indexes(data)

    # copy the dataframe and remove outliners

    new_data = data.copy(True)
    new_data = new_data.drop(labels=outliners, axis=0)
    return new_data

