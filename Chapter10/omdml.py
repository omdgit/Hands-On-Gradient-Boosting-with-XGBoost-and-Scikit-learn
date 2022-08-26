"""
Created on Sat Aug 8, 2022

@author: omd
First Version: 08.08.2022

This module has some functions usefule for machine learning
 - confusion matrix
 - precision-recall
 - AUROC
 - ROC AUC Score
 - Accuracy
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

__version__ = '8.8.2022'

def create_confusion_matrix(model, X, y, labels):
    """This function creates a confusion matrix

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        labels (_type_): _description_
    """
    # normalize confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for title, normalize in titles_options:
        if normalize == None:
            i = ax1
        else:
            i = ax2
        disp = metrics.ConfusionMatrixDisplay.from_estimator(
            model,
            X,
            y,
            display_labels=labels,
            cmap=plt.cm.Blues,
            normalize=normalize,
            ax=i,
        )
        disp.ax_.set_title(title)

        plt.grid(False)
        print(title)
        print(disp.confusion_matrix)

    plt.show()

def auroc_graph(model, X, y, label=None, title=None):
    """Plot Area under ROC curve

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        label (_type_, optional): _description_. Defaults to None.
    """
    fpr, tpr, thresh = metrics.roc_curve(
        y_true=y, y_score=model.predict_proba(X)[:, 1])
    y_pred = model.predict_proba(X)[:, 1]
    auc = metrics.roc_auc_score(y_true=y, y_score=y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))

    if title:
        title = 'ROC Curve ({0:}: n={1:,.0f})'.format(title, len(y))
    else:
        title = 'ROC Curve (n={0:,f})'.format(len(y))

    if label:
        label = 'AUC {0:} (area = {1:0.2f})'.format(label, auc)
    else:
        label = 'AUC model (area = {0:0.2f})'.format(auc)

    ax.plot(fpr, tpr, label=label)
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.plot([0, 1], [0, 1], color='orange', ls='--', label='Random')
    ax.legend()
    ax.grid(alpha=0.5)

# Precision-Recall Curve
def pr_curve(model, X, y, title=None):
    """Creates Precision-Recall Curve

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
    """

    y_pred = model.predict_proba(X)[:, 1]
    auprc = metrics.average_precision_score(y, y_pred)

    # Containers for true positive / false positive rates
    precision_scores = []
    recall_scores = []

    thresholds = np.linspace(0, 1, num=10)

    for i in thresholds:
        precision, recall, threshold = metrics.precision_recall_curve(y, y_pred)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # AUPRC baseline
    ratio = np.round(len(y[y==1]) / len(y), 2)
    if title:
        title_graph = '{0:s}\nPrecision/Recall Curve (AUPRC baseline: {1:0.2f})'.format(title, ratio)
    else:
        title_graph = 'Precision/Recall Curve (AUPRC baseline: {0:0.2f})'.format(ratio)

    plt.plot(recall_scores[0], precision_scores[0], color='steelblue')
    plt.xlim(0.0, 1.05), plt.ylim(np.min(precision_scores)-0.05), 1.05
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title_graph)
    plt.text(0.35, ((1-0.04+np.min(precision_scores))/2), 'AUPRC: ' + str(auprc.round(4)),
        bbox=dict(facecolor='red', alpha=0.5))
    plt.grid(True, alpha=0.5)
    plt.show()


# Performance metrics:
# Accuracy

def accuracy(y, y_score, label):
    """Provides accuracy of model for given dataset

    Args:
        y (_type_): _description_
        y_score (_type_): _description_
        label (_type_): _description_
    """
    acc = metrics.accuracy_score(y_true=y, y_pred=y_score)
    print('Accuracy {0:s}: {1:0.4f}'.format(label, acc))

# AUROC
def auc_score(model, X, y, label):
    """Provides ROC AUC score of model for given dataset

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        label (_type_): _description_
    """    
    y_pred = model.predict_proba(X)[:, 1]
    auc = metrics.roc_auc_score(y_true=y, y_score=y_pred)
    print('ROC AUC {0:s}: {1:0.4f}'.format(label, auc))

# Average Precision-Recall (or AUPRC)
