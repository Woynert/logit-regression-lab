import numpy as np
import pandas as pd
from typing import List, Dict


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


def sort_distances (distances: List[PointDistance]) -> List[PointDistance]:

    sorting_function = lambda obj: obj.distance
    distances.sort(key = sorting_function)
    return distances


def group_point_distances_by_tags(
    distances: List[PointDistance],
    tags: List[str]
  ) -> Dict[str, List[PointDistance]]:

  dist_groups: Dict[str, List[PointDistance]] = {}

  for tag in tags:
    pdList: List[PointDistance] = []

    for point in distances:
        if point.tag == tag:
            pdList.append(point)

    # save in dictionary

    dist_groups[tag] = pdList

  return dist_groups


def print_contestants_side_by_side (
        dist_groups: Dict[str, List[PointDistance]],
        tags: List[str],
        limit: int = 20
    ):

    line = "t:"
    for tag in tags:
        line = f"{line} {tag:.3f} "
    print (line)

    for i in range (min(limit, len(dist_groups[tags[0]]))):
        line = str(i) + " "

        for tag in tags:
            line = f"{line} {dist_groups[tag][i].distance:.3f} "

        print(line)
