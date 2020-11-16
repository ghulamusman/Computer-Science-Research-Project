from operator import itemgetter
from sklearn.utils import shuffle
from analytics.dataoverlap_interval import OverLapInt
from analytics.fp_acceptance_percentage import FpAcceptance
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
from dataset.low_variance_feature_removal import LowVarFeatRemoval
from dataset.outlier_removal import OutLierRemoval
import pandas as pd
from sklearn import inspection
from sklearn.feature_selection import mutual_info_classif

root_path = Path(__file__).parent.parent.parent.parent
gr1_feature_path = os.path.join(root_path, 'experiment_results/mouse_overlap_C_SVM/gr1_df_scaled.csv')
gr2_feature_path = os.path.join(root_path, 'experiment_results/mouse_overlap_C_SVM/gr2_df_scaled.csv')
data_metric_save_path = os.path.join(root_path, 'experiment_results/MOUSE_overlap_feature_imp_medoverlap_40914/')
gr2_per_dim_overlap_path = os.path.join(root_path,
                                        'experiment_results/mouse_overlap_C_SVM'
                                        '/gr2_hyper_vol_size_overlap_per_dim'
                                        '.csv')

rand_state = 48
# neg_sample_sources_range = list(range(neg_sample_sources_base, neg_sample_sources_base * 2, neg_sample_sources_base))
neg_sample_sources = 23
cv = 10
n_iter = 30
scoring_metric = 'precision'
cut_off = 0.1
std_dev_gr2 = 15

"""
Reading feature form disk
"""
df_group_1 = pd.read_csv(gr1_feature_path)
df_group_2 = pd.read_csv(gr2_feature_path)
"""
    Reading ovrelap data from disk and finding best seed user 
"""
gr2_per_dim_overlap = pd.read_csv(gr2_per_dim_overlap_path)
gr2_mins = gr2_per_dim_overlap.min()
gr2_mins = gr2_mins[gr2_mins < cut_off]

query_list = []
for col in gr2_mins.index:
    query_list.append(f"{col} >= {cut_off}")
query = ' & '.join(query_list)

if len(query) != 0:
    fil_gr2_overlap = gr2_per_dim_overlap.query(query)
else:
    fil_gr2_overlap = gr2_per_dim_overlap

seed_ol_user_dict = dict()
users_group_2 = df_group_2.user.unique()
for seed_user in users_group_2:
    fil_seed_user_pd = fil_gr2_overlap[(fil_gr2_overlap['V1'] == seed_user) | (fil_gr2_overlap['V1'] == seed_user)]
    seed_arr = fil_seed_user_pd.V1.unique().tolist()
    seed_arr.extend(fil_seed_user_pd.V2.unique().tolist())
    seed_arr = np.array(seed_arr)
    seed_arr = np.unique(seed_arr).tolist()
    seed_arr.remove(seed_user)
    seed_ol_user_dict[seed_user] = seed_arr

gr2_len_tup = [(key, len(seed_ol_user_dict[key])) for key in seed_ol_user_dict.keys()]
gr2_len_tup = sorted(gr2_len_tup, key=itemgetter(1), reverse=True)
best_seed_user = gr2_len_tup[0][0]

"""
Removing features with low variance
"""
df = LowVarFeatRemoval().operate(data=df_group_1)
df_group_2 = LowVarFeatRemoval().operate(data=df_group_2)

"""
Biometrics module usage example

"""
tb_data = BioDataSet(feature_data_frame=df)  # example of using data frame
tb_data_group_2 = BioDataSet(feature_data_frame=df_group_2, random_state=rand_state)
''' 
   get the user list from the dataset class object
'''

users = tb_data.user_list
users_group_2 = tb_data_group_2.user_list
''' 
   generate tagged data set for each user
'''
Data = dict()
data_group_2 = dict()

for user in users:
    Data[user] = tb_data.get_data_set(user, neg_sample_sources=neg_sample_sources, neg_test_limit=True)

for user in users_group_2:
    data_group_2[user] = df_group_2[df_group_2['user'] == user]

"""
    Extracting overlapping features from group 2
"""
seed_user_gr2 = best_seed_user

overlap_data_gr_2_489146 = data_group_2[seed_user_gr2]


for user in seed_ol_user_dict[seed_user_gr2]:
    overlap_data_gr_2_489146, df_2 = \
        OverLapInt(overlap_data_gr_2_489146, data_group_2[user], std_dev=std_dev_gr2).get_analytics()


overlap_data_gr_2 = overlap_data_gr_2_489146
overlap_data_gr_2['labels'] = np.zeros(len(overlap_data_gr_2))

"""
Overlap values in both x and y:
Most                       Smallest
53138 0.89  99401 0.87   40914 0.80

"""

pos_user = 40914

data_frame = Data[pos_user]
data_frame.to_csv(os.path.join(data_metric_save_path, f"{pos_user}_user_pop_df.csv"), index=False, mode='w+')

pos_val_for_overlap_data = data_frame[data_frame['user'] == pos_user].iloc[0:len(overlap_data_gr_2), :]
overlap_data_gr_2 = overlap_data_gr_2.append(pos_val_for_overlap_data)
overlap_data_gr_2 = shuffle(overlap_data_gr_2)
overlap_data_gr_2 = overlap_data_gr_2.reset_index(drop=True)

feature_names = list(data_frame.columns.drop(['user', 'labels']))

''' 
  Initialize classifier object and split the data into training, testing and tuning data
'''
clf_rf = RandomForestClassifier(pos_user=pos_user, random_state=rand_state)
clf_rf.split_data(data_frame=data_frame, training_data_size=0.6, save_path=data_metric_save_path)

clf_svm = SvmClassifier(pos_user=pos_user, random_state=rand_state)
clf_svm.split_data(data_frame=data_frame, training_data_size=0.6, save_path=data_metric_save_path)
"""
   Test set and labels extraction
"""
test_set_rf = clf_rf.test_data_frame.drop('labels', axis=1)
test_labels_rf = clf_rf.test_data_frame.labels.values

train_set_rf = clf_rf.training_data_frame.drop('labels', axis=1)
train_labels_rf = clf_rf.training_data_frame.labels.values

test_set_svm = clf_svm.test_data_frame.drop('labels', axis=1)
test_labels_svm = clf_svm.test_data_frame.labels.values

train_set_svm = clf_svm.training_data_frame.drop('labels', axis=1)
train_labels_svm = clf_svm.training_data_frame.labels.values

overlap_data_gr_2_p = overlap_data_gr_2.drop('user', axis=1)
overlap_set_values = overlap_data_gr_2_p.drop('labels', axis=1)
overlap_labels = overlap_data_gr_2_p.labels.values


criterion = ['entropy']
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
grid_rf = {'criterion': criterion,
           'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           'bootstrap': bootstrap}

''' 
   Model parameter tuning using random search with cross validation and finding important features
'''

clf_rf.random_train_tune_parametres(pram_dist=grid_rf, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)
rf_important_features = pd.DataFrame(clf_rf.classifier.feature_importances_, index=feature_names)
rf_important_features = rf_important_features.sort_values(by=0)

rf_important_features.to_csv(os.path.join(data_metric_save_path,
                                          f"{pos_user}_rf_feat_rank_ent.csv"), index=True, mode='w+')

rf_important_features_perm_imp = inspection.permutation_importance(estimator=clf_rf.classifier,
                                                                   X=clf_rf.test_data_frame.drop('labels', axis=1),
                                                                   y=clf_rf.test_data_frame.labels.values,
                                                                   n_jobs=-1, n_repeats=10, random_state=rand_state)
rf_important_features_perm_mean = pd.DataFrame(rf_important_features_perm_imp.importances_mean, index=feature_names)
rf_important_features_perm_mean = rf_important_features_perm_mean.sort_values(by=0)
rf_important_features_perm_mean.to_csv(os.path.join(data_metric_save_path,
                                                    f"{pos_user}_rf_perm_imp_feat_rank_ent.csv"), index=True,
                                       mode='w+')

rf_mutual_info_imp_feat = mutual_info_classif(train_set_rf, train_labels_rf, random_state=rand_state)
rf_mutual_info_imp_feat = pd.DataFrame(rf_mutual_info_imp_feat, index=feature_names)
rf_mutual_info_imp_feat = rf_mutual_info_imp_feat.sort_values(by=0)
rf_mutual_info_imp_feat.to_csv(os.path.join(data_metric_save_path,
                                            f"{pos_user}_rf_mut_inf_feat_rank_ent.csv"), index=True, mode='w+')

''' 
   SVM Model model tuning using random search cv
'''
# c_range = np.logspace(-2, 6, 100)
c_range = [4534.878508128591]
gamma_range = np.logspace(-9, 3, 13)
pram_grid = {'C': c_range}

clf_svm.random_train_tune_parameters(pram_dist=pram_grid, cv=cv, scoring_metric=scoring_metric, n_itr=n_iter)

svm_important_features_perm = inspection.permutation_importance(estimator=clf_svm.classifier,
                                                                X=clf_svm.test_data_frame.drop('labels', axis=1),
                                                                y=clf_svm.test_data_frame.labels.values,
                                                                n_jobs=-1, n_repeats=10, random_state=rand_state)
svm_important_features_perm_mean = pd.DataFrame(svm_important_features_perm.importances_mean, index=feature_names)
svm_important_features_perm_mean = svm_important_features_perm_mean.sort_values(by=0)
svm_important_features_perm_mean.to_csv(os.path.join(data_metric_save_path,
                                                     f"{pos_user}_svm_perm_imp_feat_rank_ent.csv"), index=True,
                                        mode='w+')

svm_mutual_info_imp_feat = mutual_info_classif(train_set_svm, train_labels_svm, random_state=rand_state)
svm_mutual_info_imp_feat = pd.DataFrame(svm_mutual_info_imp_feat, index=feature_names)
svm_mutual_info_imp_feat = svm_mutual_info_imp_feat.sort_values(by=0)
svm_mutual_info_imp_feat.to_csv(os.path.join(data_metric_save_path,
                                             f"{pos_user}_svm_mut_inf_feat_rank_ent.csv"), index=True, mode='w+')

''' 
   Model Classification
'''

predictions_rf = clf_rf.classify()
predictions_svm = clf_svm.classify()

overlap_dataset_predictions_rf = clf_rf.classify(df=overlap_data_gr_2_p)
overlap_dataset_predictions_svm = clf_svm.classify(df=overlap_data_gr_2_p)

"""
    False positive acceptance percentage example
"""
fp_pers_rf = FpAcceptance(df=overlap_data_gr_2_p, prediction=clf_rf.predictions_ext_df)
percent_overlap_sample_accept_rf = round(fp_pers_rf.get_analytics(), 2)
print('%.2f' % percent_overlap_sample_accept_rf, "percent of false users accepted for RF classifier")

fp_pers_svm = FpAcceptance(df=overlap_data_gr_2_p, prediction=clf_svm.predictions_ext_df)
percent_overlap_sample_accept_svm = round(fp_pers_svm.get_analytics(), 2)
print('%.2f' % percent_overlap_sample_accept_svm, "percent of false users accepted for SVM classifier")

"""
    Confusion Matrix Curves
"""

rf_cm_path = os.path.join(data_metric_save_path, f'{pos_user}_rf_cm.csv')
cm_rf = ConfusionMatrix()
matrix_rf = cm_rf.get_metric(true_labels=test_labels_rf, predicted_labels=predictions_rf, output_path=rf_cm_path)

svm_cm_path = os.path.join(data_metric_save_path, f'{pos_user}_svm_cm.csv')
cm_svm = ConfusionMatrix()
matrix_svm = cm_svm.get_metric(true_labels=test_labels_svm, predicted_labels=predictions_svm, output_path=svm_cm_path)

"""
        Confusion Overlap Set, FpAcceptance calculates confusion matrix internally
"""

overlap_cm = fp_pers_svm.cm

"""
    ROC Curves
"""
ax_roc = plt.gca()

roc_rf = RocCurve().get_metric(test_set_features=test_set_rf.values, test_set_labels=test_labels_rf
                               , classifier=clf_rf.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, f'{pos_user}_ROC_RF.png'), )

roc_svm = RocCurve().get_metric(test_set_features=test_set_svm.values, test_set_labels=test_labels_svm
                                , classifier=clf_svm.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, f'{pos_user}_ROC_SVM.png'))

ax_roc.figure.set_figheight(12)

overlap_roc_rf = RocCurve().get_metric(test_set_features=overlap_set_values.values, test_set_labels=overlap_labels
                                       , classifier=clf_rf.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, f'{pos_user}_OL_ROC_RF.png'))

overlap_roc_svm = RocCurve().get_metric(test_set_features=overlap_set_values.values, test_set_labels=overlap_labels
                                        , classifier=clf_svm.classifier, ax=ax_roc)
plt.savefig(os.path.join(data_metric_save_path, f'{pos_user}_OL_ROC_SVM.png'))

ax_roc.figure.set_figwidth(12)
ax_roc.figure.savefig((os.path.join(data_metric_save_path, f'{pos_user}_ROC_SVM_RF.png')))

"""
    FCS
"""
fcs_rf = FCS(classifier_name='RF')
fcs_rf.get_metric(true_labels=test_labels_rf, predicted_probs=clf_rf.predictions_prob, pred_labels=clf_rf.predictions)
plt.savefig(os.path.join(data_metric_save_path, f'{pos_user}_FCS_RF.png'))

fcs_svm = FCS(classifier_name='svc')
fcs_svm.get_metric(true_labels=test_labels_svm, predicted_probs=clf_svm.predictions_prob,
                   pred_labels=clf_svm.predictions)
plt.savefig(os.path.join(data_metric_save_path, f'{pos_user}_FCS_SVM.png'))

"""
Calculating mean overlaps
"""
pos_user_per_dim_ol_path = os.path.join(data_metric_save_path, "hyper_vol_size_overlap_per_dim.csv")
pos_user_per_dim_ol = pd.read_csv(pos_user_per_dim_ol_path)
pos_user_per_dim_ol = pos_user_per_dim_ol.drop('Unnamed: 0', axis=1)

pos_user_pd_ol_others = pos_user_per_dim_ol[(pos_user_per_dim_ol['V1'] == pos_user)]

pos_user_pd_ol_others.drop(['V1', 'V2'], axis=1).mean().sort_values(ascending=False).to_csv(
    os.path.join(data_metric_save_path, f"{pos_user}_mean_ol_others_per_dim.csv"), index=True, mode='w+')

pos_user_pd_ol_by_others = pos_user_per_dim_ol[(pos_user_per_dim_ol['V2'] == pos_user)]
pos_user_pd_ol_by_others.drop(['V1', 'V2'], axis=1).mean().sort_values(ascending=False).to_csv(
    os.path.join(data_metric_save_path, f"{pos_user}_mean_ol_by_others_per_dim.csv"), index=True, mode='w+')


plt.show()