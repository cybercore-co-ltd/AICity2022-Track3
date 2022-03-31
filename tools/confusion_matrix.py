import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from mmaction.core import confusion_matrix

if __name__=='__main__':
    
    pred = np.load("pred.npy")
    labels = np.load("labels.npy")
    
    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    
    cf_mat = confusion_matrix(pred, labels).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sns.heatmap(cf_mat, annot=True, annot_kws={"size": 8}, cmap='Blues', square=True)
        
    fig.axes[0].set_xlabel('Predicted')
    fig.axes[0].set_ylabel('True')
    fig.savefig('confusion_matrix.png', dpi=250)
    plt.close()