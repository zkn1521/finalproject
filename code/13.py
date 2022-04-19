from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

# split into features and label set
X_train_set = []
X_test_set = []
X = pd.read_csv('training_set.csv')
Y = X["abnormal"]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_index, test_index in split.split(X, X["abnormal"]):
    X_train_set = X.loc[train_index]
    X_test_set = X.loc[test_index]  # 保证测试集

print(X_train_set)
print(X_test_set)
X_train_set.to_csv('X_training_set_2_labels_second_version.csv', encoding='utf-8', index=False,
                   columns=['self_citing_rate', 'before_two_years_self_citing_rate', 'abnormal',
                            'average_pub_rate',
                            'Reference_count_of_each_paper_published',
                            'two_years_incoming_citations', 'two_years_outgoing_references'])
X_train_set.to_csv('X_training_set_2_labels_label_second_version.csv', encoding='utf-8', index=False,
                   columns=['abnormal'])
X_test_set.to_csv('X_test_set_2_labels_second_version.csv', encoding='utf-8', index=False,
                  columns=['self_citing_rate', 'before_two_years_self_citing_rate', 'average_pub_rate',
                           'Reference_count_of_each_paper_published',
                           'two_years_incoming_citations', 'two_years_outgoing_references'])
X_test_set.to_csv('X_test_set_2_labels_label_second_version.csv', encoding='utf-8', index=False,
                  columns=['abnormal'])
