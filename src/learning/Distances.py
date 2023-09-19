import numpy as np
import pandas as pd
from typing import List


class PointDistance:
    index: int = -1
    distance: float = 0
    tag: str = ''

    def __init__(self, index, distance, tag):
        self.index = index
        self.distance = distance
        self.tag = tag

    def __str__(self):
        return f'{self.index} {self.distance} {self.tag}'

    def print(self):
        print(str(self))


def point_distances_against_others (
        point:           pd.DataFrame, # target point
        data_src:        pd.DataFrame, # dataset
        tag_column:      str,          # column containing the tag
        exclude_columns: List[str]     # columns to exclude
    ) -> List[PointDistance]:

    distances: List[PointDistance] = []

    # drop unwanted columns

    data_tmp = data_src.drop(exclude_columns, axis=1) 

    for i in range(len(data_tmp.index)):
        row = pd.DataFrame([data_tmp.iloc[i]])

        # get row tag
        tag = (pd.DataFrame([data_src.iloc[i]])).iloc[0][tag_column]

        # compare
        acum_cubes = 0
        for column in row:
            value_current_point = row.iloc[0][column]
            value_target_point = point.iloc[0][column]
            acum_cubes = acum_cubes + (value_target_point - value_current_point) ** 2

        # store
        distances.append(PointDistance(i, np.sqrt(acum_cubes), tag))

    return distances
