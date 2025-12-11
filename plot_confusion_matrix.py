import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

rf_full = np.array([[10204,     5,     0], [5,   177,     2],[0,  2,   258]])
xgb_full =  np.array([[10209,     0,     0],[0,   180,     4],[0,     2,   258]])
nn_full =  np.array([[10202,     7,     0],[0,   174,    10],[ 0,    11,   249]])

rf_noEarly =  np.array([[10208,     1,     0],[0,   180,    4],[0,     0,   260]])
xgb_noEarly =  np.array([[10209,     0,     0],
 [    0,   180,     4],
 [    0,     2,   258]])
nn_noEarly = np.array([[10208,     1,     0],
 [    0,   181,     3],
 [    0,    19,   241]])

rf_Early20 =  np.array([[10142,    61,     6],
 [   16,   155,    13],
 [   11,     9,   240]])
xgb_Early20 =  np.array([[10114,    88,     7],
 [   17,   156,    11],
 [   10,    47,   203]])
nn_Early20 =  np.array([[10069,   124,    16],
 [   14,   148,    22],
 [    0,     1,   259]])

rf_Early30 = np.array([[10141,    62,     6],
 [   20,   150,    14],
 [    9,    11,   240]])
xgb_Early30 =  np.array([[10126,    79,     4],
 [   19,   155,    10],
 [    3,    53,   204]])
nn_Early30 =  np.array([[10106,    89,    14],
 [   27,   140,   17],
 [    0,    7,   253]])

rf_Early50 =  np.array([[10138,    65,     6],
 [   22,   149,    13],
 [    9,    11,   240]])
xgb_Early50 =  np.array([[10122,  83,     4],
 [   21,   153,    10],
 [    0,    35,   225]])
nn_Early50 =  np.array([[10117,    76,    16],
 [   21,   136,    27],
 [    0,     0,   260]])

rf_Early60 = np.array([[10130,    73,     6],
 [   22,   149,    13],
 [    0,     6,   254]])
xgb_Early60 = np.array([[10129,    76,     4],
 [   23,   151,    10],
 [    3,    31,   226]])
nn_Early60 = np.array([[10116,    77,    16],
 [   23,   131,    30],
 [    0,     0,   260]])


def plot_cm(cm, title, save_path=None):
    # cm = confusion_matrix(y_true, y_pred, labels=labels)
    # cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]) * 100

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=['bad', 'good_no_drone', 'good_drone'],
                yticklabels=['bad', 'good_no_drone', 'good_drone'],
                linecolor='black', linewidths=2,
                cmap="Blues")
    plt.title(title + " (normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


plot_cm(cm=rf_Early30,
    title="RF - Early30",
    save_path="plots_out/cm_rf_Early30.png"
)


