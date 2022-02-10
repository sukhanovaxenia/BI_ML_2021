import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    n_sample = y_pred.shape[0]
    TP, FP = 0, 0
    TN, FN = 0, 0
    test = y_true.astype(int)
    for i in range(n_sample):
        if test[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif test[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif test[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif test[i] == 1 and y_pred[i] == 0:
            FN += 1
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        f_m = 2*(precision*recall/(precision + recall))
    except ZeroDivisionError:
        print("Could not divide by zero. Precision and F1 score can't be counted")
        recall = TP/(TP+FN)
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        precision, f_m = None, None
        
    
    return precision, recall, accuracy, f_m



def multiclass_accuracy(y_pred, y_test, adj = False):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    
    Clues:
    accuracy = (TP+TN)/N can be implemented as number of Trues
                                from array-to-array comparison
    if adjacent accuracy has to be counted (including deviation)
                                    - adj = True should be set
    
    """
    n_sample = y_pred.shape[0]
    total_accuracy = np.sum(y_pred == y_test.astype(int))/n_sample
    if adj:
        #if adj = T deviation is counted
        adj_acc = sum(abs(y_pred - y_test.astype(int)) <= 1)/n_sample
        return adj_acc
    return total_accuracy


def r_squared(predict_Di, y_arr_tst):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r_sq_up, r_sq_down = 0, 0
    for i in range(len(predict_Di)):
        r_sq_up += ((predict_Di[i] - y_arr_tst[i]) ** 2)
        r_sq_down += ((predict_Di[i] - float(predict_Di.mean())) ** 2)

    return (1 - r_sq_up/r_sq_down)


def mse(predict_Di, y_arr_tst):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    mse = 0
    for i in range(len(predict_Di)):
        mse += ((predict_Di[i] - y_arr_tst[i]) ** 2)
    return mse/float(len(X_arr_tst))


def mae(predict_Di, y_arr_tst):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    mae = 0
    for i in range(len(predict_Di)):
        mae += abs(predict_Di[i] - y_arr_tst[i])
    return mae/float(len(X_arr_tst))

    