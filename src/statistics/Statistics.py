import pandas as pd
from typing import List, Tuple, Dict
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


def calculate_confusion_matrix(
        k: int,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        tags: List[str],
        tag_column: str,
        tag_objective: List[str]
    ) -> Dict[str, int]:
    
    confusion_matrix = {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0
    }
    
    for i in range(len(data_test.index)):
    
        # get point
        point = pd.DataFrame([data_test.iloc[i]])
        #print(point)
        
        # predict using model
        predicted_tag = Categorization.categorize_point(
            data_train,
            point,
            k,  # k
            tags,
            tag_column
        )
        
        # Etiqueta real
        actual_tag = point.iloc[0][tag_column]
        
        #print(predicted_tag, ":", actual_tag)
        # resultado de interes
        if predicted_tag in tag_objective:
        
            # clasificacion correcta
            if predicted_tag == actual_tag:
                confusion_matrix['TP'] += 1
            else:
                confusion_matrix['FP'] += 1
        
        # no nos dio el valor de interes
        else:
            # clasificacion correcta
            if predicted_tag == actual_tag:
                confusion_matrix['TN'] += 1
            else:
                confusion_matrix['FN'] += 1
    
    return confusion_matrix 


def get_roc_data_from_model (
        k_list: List[int],
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        tags: List[str],
        tag_column: str,
        tag_objective: List[str]
    ) -> Tuple[List[int], List[int]]:

    listTPR = []
    listFPR = []
    
    for k in k_list:

        matrix = calculate_confusion_matrix(
            k,
            data_train,
            data_test,
            tags,
            tag_column,
            tag_objective
        )

        listTPR.append(matrix['TP'] / (matrix['TP'] + matrix['FN']))
        listFPR.append(matrix['FP'] / (matrix['FP'] + matrix['TN']))

    return listTPR, listFPR
