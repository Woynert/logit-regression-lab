from typing import List
import numpy as np
import pandas as pd
import random

SENS = 1.5

def is_outliner (value, q1, q3, iqr) -> bool:

    return (value < (q1 - SENS * iqr)) or (value > (q3 + SENS * iqr))

def get_outliner_indexes (data: pd.DataFrame) -> List[int]:
    
    outliner_indexes: List[int] = []

    for column in data:

        vlist = data[str(column)].tolist()
        q1 = np.percentile(vlist, 25)
        q3 = np.percentile(vlist, 75)
        iqr = q3 - q1

        for i in data[str(column)].index:

            v = data[str(column)][i]
        
            if is_outliner (v, q1, q3, iqr):
                outliner_indexes.append(i)

    return list (set (outliner_indexes))   
    

def remove_outliners (data: pd.DataFrame) -> pd.DataFrame:

    outliners: List[int] = get_outliners_indexes(data)

    # copy the dataframe and remove outliners

    new_data = data.copy(True)
    new_data = new_data.drop(labels=outliners, axis=0)
    return new_data


def delete_random_amount(data: pd.DataFrame
    ) -> pd.DataFrame:
    
    outliners: List[int] = get_outliner_indexes(data)
    
    new_data = data.copy(True)
    outliners_mixed = []
    
    d = int(20*len(data.index)/100)
    
    if len(outliners)>(40*len(data.index)/100):
        outliners_mixed = random.sample(outliners,d)
        
        new_data = new_data.drop(labels=outliners_mixed, axis=0)
    return new_data


 

