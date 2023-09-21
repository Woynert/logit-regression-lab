import pandas as pd
from typing import List
import Categorization


# categorize all test points and calculate MAPE

def calculate_mape(
        k: int,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        tag_column: str,
        tags: List[str]
    ) -> int:

    mape = 0

    for i in range (len(data_test.index)):

        # get point
        point = pd.DataFrame([data_test.iloc[i]])

        # predict using model
        tag = Categorization.categorize_point(
            data_train,
            point,
            k, # k
            tags,
            tag_column
        )

        # check the prediction is incorrect
        if (tag != point.iloc[0][tag_column]):
            mape += 1

    return (mape / len(data_test.index))
