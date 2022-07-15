import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Orange.evaluation import compute_CD, graph_ranks
from scipy.stats import friedmanchisquare, rankdata


auc_orange = [0.57, 0.86, 0.73, 0.96, 0.98, 0.98, 0.90, 0.67, 0.61, 0.70, 0.71]
auc_xgb = [0.33, 0.85, 0.72, 0.94, 0.98, 0.98, 0.86, 0.70, 0.57, 0.69, 0.70]
auc_rf = [0.41, 0.79, 0.66, 0.94, 0.98, 0.98, 0.89, 0.69, 0.61, 0.70, 0.63]
auc_lr = [0.57, 0.86, 0.73, 0.92, 0.94, 0.96, 0.87, 0.65, 0.59, 0.69, 0.67]
auc_svm = [0.49, 0.82, 0.72, 0.87, 0.95, 0.96, 0.87, 0.63, 0.55, 0.70, 0.66]

f1_lce = [0.88, 0.963, 0.86435, 0.5723, 0.882, 0.4795,0.752]
f1_xgb = [0.8392,0.8758, 0.842304,0.31295, 0.7576,0.5867,0.857]
f1_rf = [0.869956,0.97, 0.8637,0.4536, 0.8172,0.5889,0.63]
f1_svm = [0.8392,0.8758, 0.858977,0.31295, 0.58,0.5867,0.848]

# #sign test AUC
# performances = pd.DataFrame({'dataset':['df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'df9', 'df10', 'df11'],'ORANGE': auc_orange, 'XGB': auc_xgb, 'RF': auc_rf, 'SVM': auc_svm, 'LR': auc_lr})
# sign test F1-score
performances = pd.DataFrame({'dataset':['df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7'],'LCE': f1_lce, 'XGB': f1_xgb, 'RF': f1_rf, 'SVM': f1_svm})

# First, we extract the algorithms names.
algorithms_names = performances.drop('dataset', axis=1).columns

print(algorithms_names)
# Then, we extract the performances as a numpy.ndarray.
performances_array = performances[algorithms_names].values
print(performances_array)
# Finally, we apply the Friedman test.
print(friedmanchisquare(*performances_array))
ranks = np.array([rankdata(-p) for p in performances_array])

# Calculating the average ranks.
average_ranks=np.mean(ranks,axis=0)
print('\n'.join('{} average rank: {}'.format(a,r) for a,r in zip(algorithms_names, average_ranks)))

cd = compute_CD(average_ranks, n=len(performances), alpha='0.05', test='nemenyi')
# This method generates the plot.
graph_ranks(average_ranks, names=algorithms_names, cd=cd, width=12, textspace=3, reverse=True)
plt.savefig('test.png',dpi=300)
plt.show()
