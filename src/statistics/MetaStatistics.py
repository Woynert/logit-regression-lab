import pandas as pd
from typing import List, Tuple, Dict
import Categorization
import matplotlib.pyplot as plt


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
        columns: List[str],
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

        # exclude columns
        all_columns = [str(c) for c in data_train]
        exclude_columns = [c for c in all_columns if c not in columns]
        
        # predict using model
        predicted_tag = Categorization.categorize_point(
            data_train,
            point,
            k,  # k
            tags,
            tag_column,
            exclude_columns
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


def get_multiple_confusions_per_model (
        k_list: List[int],
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        exclude_columns: List[str],
        tags: List[str],
        tag_column: str,
        tag_objective: List[str]
    ) -> List[Dict[str, int]]:

    mlist = []
    
    for k in k_list:

        matrix = calculate_confusion_matrix(
            k,
            data_train,
            data_test,
            exclude_columns,
            tags,
            tag_column,
            tag_objective
        )

        mlist.append(matrix)

    return mlist


def get_roc_data_from_model (
        matrixs: List[Dict[str, int]]
    ) -> Tuple[List[int], List[int]]:

    listTPR = []
    listFPR = []

    for matrix in matrixs:
        listTPR.append(matrix['TP'] / (matrix['TP'] + matrix['FN']))
        listFPR.append(matrix['FP'] / (matrix['FP'] + matrix['TN']))

    return listTPR, listFPR


# sensibilidad, especificidad y precisión

def find_performance (
        matrix: Dict[str, int],
        i: int = -1
    ):

    sensi = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    espec = matrix['TN'] / (matrix['FP'] + matrix['TN'])
    preci = matrix['TP'] / (matrix['TP'] + matrix['FP'])

    print(f"{i}. sen: {sensi:.5f}, esp: {espec:.5f}, pre: {preci:.5f}")


# meta function for testing different models

def model_meta_function(
        k_list: List[int],
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        columns: List[str],
        tags: List[str],
        tag_column: str,
        tag_objective: List[str]
    ) -> Dict[str, int]:

    # sacar metricas

    listMatrix = get_multiple_confusions_per_model (
        k_list,
        data_train,
        data_test,
        columns,
        tags,
        tag_column,
        tag_objective
    )

    for matrix in listMatrix:
        
        print (matrix)
        
    # sacar performance
        
    for matrix in listMatrix:
        
        find_performance(matrix)

    # sacar curva ROC

    lTPR, lFPR = get_roc_data_from_model(listMatrix)

    slTPR, slFPR = zip(*sorted(zip(lTPR, lFPR)))

    for i in range(len(lTPR)):
        print (slFPR[i], ":", slTPR[i])
    
    #create ROC curve
    plt.plot(slFPR, slTPR)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
        