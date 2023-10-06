import pandas as pd
from typing import Dict, List
from Distances import (
    PointDistance,
    group_point_distances_by_tags,
    point_distances_against_others,
    sort_distances,
)

# compare tag distance columns parallelly
# Returns winner tag

def compare_parallelly(
    dist_groups: Dict[str, List[PointDistance]],
    tags:        List[str],
    k:           int # hyperparameter
  ) -> str:

    votes: Dict[str, int] = {}

    for tag in tags:
        votes[tag] = 0

    # vote

    for i in range(k):

        winner_dist = float('inf')
        winner = ''

        for tag in tags:

            dist = dist_groups[tag][i].distance

            if dist < winner_dist:
                winner_dist = dist
                winner = tag

        #print("winner of round", i, "is", winner)
        votes[winner] = votes[winner] + 1

    # get winner

    winner_votes: int = 0
    winner_tag: str = ''

    for tag in tags:
        if votes[tag] > winner_votes:
            winner_votes = votes[tag]
            winner_tag = tag

    return winner_tag


def categorize_point(
        data:  pd.DataFrame,
        point: pd.DataFrame,
        k:     int,          # hyper parameter
        tags:  List[str],
        tag_column:      str,
        exclude_columns: List[str] = [] # columns to exclude
    ) -> str:

    # calculate point distances
    distances: List[PointDistance] = \
        point_distances_against_others (point, data, tag_column, exclude_columns)

    # sort
    sorted_distances: List[PointDistance] = sort_distances(distances)

    # separate sorted distances by column
    dist_groups: Dict[str, List[PointDistance]] = \
        group_point_distances_by_tags(sorted_distances, tags)

    # determine knn winner by votes
    return compare_parallelly(dist_groups, tags, k)
