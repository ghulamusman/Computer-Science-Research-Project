from classifiers.svm_classifier import SvmClassifier
from dataset.min_max_scaling_operation import MinMaxScaling
from dataset.biometric_dataset import BioDataSet
from external_dataset_parsers import dsn_keystroke_parser
from metrics.f1_score import F1Score
from metrics.roc_curve import RocCurve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from external_dataset_parsers import hmog_parser
from dataset.biometric_dataset import BioDataSet
from dataset.min_max_scaling_operation import MinMaxScaling
from dataset.standard_scaling_operation import StandardScaling
from dataset.dim_red_pca_operation import PcaDimRed
from classifiers.random_forest_classifier import RandomForestClassifier
from classifiers.knn_classifier import KnnClassifier
from classifiers.svm_classifier import SvmClassifier
import numpy as np
from metrics.confusion_matrix import ConfusionMatrix
from metrics.gini_coefficient import GiniCoef
from metrics.fcs import FCS
from metrics.roc_curve import RocCurve
import matplotlib.pyplot as plt
from pathlib import Path
import os
from external_dataset_parsers.mouseparser import MouseParser

from dataset.low_variance_feature_removal import LowVarFeatRemoval

"""
Example of scm classifier operation with parameter tuning
"""

''' 
   Read the processed feature data from disk and generate features
'''
root_path = Path(__file__).parent.parent.parent.parent
print(root_path)
processed_data = os.path.join(root_path, 'processed_data/dsn_keystroke/df')

df = dsn_keystroke_parser.DSNParser().get_feature_vectors(processed_data, limit=None)
df['user'] = df['user'].apply(lambda x: int(x[1:]))
''' 
   Read the processed feature data from disk and generate features
'''

# use raw_to_feature_vectors for raw data with 1 for choice for mouse else for trackpad
scaled_data_path_group_1 = os.path.join(root_path,
                                        'experiment_results/4900/df_scaled.csv')

data_metric_save_path = os.path.join(root_path,
                                     'experiment_results/4900')

tb_data = BioDataSet(feature_data_frame=df)
users = tb_data.get_users_list
print(len(users))

subset50 = df.loc[df['user'].isin(users[:50])]

rand_state = 42
neg_sample_sources = 10
cv = 30
n_iter = 30
scoring_metric = 'precision'
test_train_split_ratio = 0.5

df = subset50
Data = dict()
for user in users:
    Data[user] = tb_data.get_data_set(user, neg_sample_sources=neg_sample_sources, neg_test_limit=True)

''' 
   perform min max scaling
'''

min_max_tuple = (0, 1)
MinMaxData = dict()
for user in Data:
    MinMaxData[user] = MinMaxScaling().operate(Data[user], min_max_tuple)


''' 
    Classifier module usage example
   Initialize classifier object and split the data into training, testing and tuning data
'''

data_frame = MinMaxData[users[0]]
data_frame.to_csv(os.path.join(data_metric_save_path, "user_pop_df.csv"), index=False, mode='w+')
pos_user = users[0]
clf_rf = RandomForestClassifier(pos_user=pos_user, random_state=rand_state)
clf_rf.split_data(data_frame=data_frame, training_data_size=test_train_split_ratio, save_path=data_metric_save_path)

clf_svm = SvmClassifier(pos_user=pos_user, random_state=rand_state)
clf_svm.split_data(data_frame=data_frame, training_data_size=test_train_split_ratio, save_path=data_metric_save_path)

clf_knn = KnnClassifier(pos_user=pos_user, random_state=rand_state)
clf_knn.split_data(data_frame=data_frame, training_data_size=test_train_split_ratio, save_path=data_metric_save_path)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
grid_rf = {'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           'bootstrap': bootstrap}

clf_rf.random_train_tune_parametres(pram_dist=grid_rf, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

''' 
   SVM MODEL PARAMETER TUNING
'''

c_range = range(1, 21)
grid_svm = {'C': c_range}

clf_svm.train_tune_parameters(pram_grid=grid_svm, cv=cv, scoring_metric=scoring_metric)

"""
KNN model tuning using random search cv
"""
leaf_size = list(range(1, 70))
n_neighbors = list(range(1, 50))
p = [1, 2]

grid_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

clf_knn.random_train_tune_parameters(pram_dist=grid_knn, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

''' 
   Model Classification
'''
predictions_rf = clf_rf.classify()
predictions_svm = clf_svm.classify()
predictions_knn = clf_knn.classify()

"""
    Metrics module example
"""

"""
    ROC Curves
"""
"""
    RF
"""
test_set_rf = clf_rf.test_data_frame.drop('labels', axis=1)
test_labels_rf = clf_rf.test_data_frame.labels.values

ax = plt.gca()
roc_rf = RocCurve().get_metric(test_set_features=test_set_rf.values, test_set_labels=test_labels_rf
                               , classifier=clf_rf.classifier, ax=ax)
rf_cm_path = os.path.join(data_metric_save_path, 'rf_cm.csv')
cm_rf = ConfusionMatrix()
matrix_rf = cm_rf.get_metric(true_labels=test_labels_rf, predicted_labels=predictions_rf, output_path=rf_cm_path)

plt.savefig(os.path.join(data_metric_save_path, 'ROC_RF.png'))
plt.close()

"""
    SVM
"""
test_set_svm = clf_svm.test_data_frame.drop('labels', axis=1)
test_labels_svm = clf_svm.test_data_frame.labels.values

roc_svm = RocCurve().get_metric(test_set_features=test_set_svm.values, test_set_labels=test_labels_svm
                                , classifier=clf_svm.classifier, ax=ax)
svm_cm_path = os.path.join(data_metric_save_path, 'svm_cm.csv')
cm_svm = ConfusionMatrix()
matrix_svm = cm_svm.get_metric(true_labels=test_labels_svm, predicted_labels=predictions_svm,
                               output_path=svm_cm_path)
plt.savefig(os.path.join(data_metric_save_path, 'ROC_SVM.png'))
plt.close()
"""
    KNN
"""

knn_cm_path = os.path.join(data_metric_save_path, 'knn_cm.csv')
test_set_knn = clf_knn.test_data_frame.drop('labels', axis=1)
test_labels_knn = clf_knn.test_data_frame.labels.values

cm_knn = ConfusionMatrix()
matrix_knn = cm_knn.get_metric(true_labels=test_labels_knn, predicted_labels=predictions_knn,
                               output_path=knn_cm_path)

roc_knn = RocCurve().get_metric(test_set_features=test_set_knn.values, test_set_labels=test_labels_knn
                                , classifier=clf_knn.classifier, ax=ax)
plt.savefig(os.path.join(data_metric_save_path, 'ROC_KNN.png'))
plt.close()

ax.figure.savefig((os.path.join(data_metric_save_path, 'ROC_SVM_RF_KNN.png')))
plt.close()
