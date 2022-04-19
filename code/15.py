from sklearn import metrics, tree, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, learning_curve, validation_curve, \
    GridSearchCV, LeaveOneOut, cross_validate
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

average = 'binary'
multi_class = 'raise'
labels = [0, 1]
target_names = ['0', '1']

pd.set_option('display.max_columns', None)
X_train = pd.read_csv('aa2_x_training_set1_normalization.csv')
Y_train = pd.read_csv('aa2_y_training_set1.csv')
X_test = pd.read_csv('aatest_nobalance_x_test_set_normalization.csv')
Y_test = pd.read_csv('aatest_nobalance_y_test_set.csv')

# Leave one out
loovc = LeaveOneOut()
loovc.get_n_splits(X_train)
# n-cross validation
cv = KFold(n_splits=10, shuffle=True, random_state=0)

# KNN
param_dist1 = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [i for i in range(10, 40)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [i for i in range(10, 40)]
    }
]

# 决策树
param_dist2 = {'splitter': ('best', 'random'),
               'criterion': ("gini", "entropy"),
               "max_depth": [*range(1, 11)],
               'min_samples_leaf': [*range(1, 11)],
               'min_samples_split': [*range(2, 20)],
               'max_features': ['auto', 'sqrt', 'log2', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               }

# 随机森林(bagging)
param_dist3 = {
    'n_estimators': [50, 120, 160, 200, 250, 500, 1000, 1250],
    'criterion': ("gini", "entropy"),
    "max_depth": [*range(2, 20, 1)],
    'min_samples_leaf': [*range(1, 30, 1)],
    'min_samples_split': [*range(2, 20, 1)],
    'max_features': ['auto', 'sqrt', 'log2', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# Gradient Tree Boosting(梯度提升决策树)
param_dist4 = {
    'n_estimators': [50, 120, 160, 200, 250, 500, 1000, 1250],
    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    "max_depth": [*range(2, 20, 1)],
    'min_samples_leaf': [*range(1, 30, 1)],
    'min_samples_split': [*range(2, 20, 1)],
    'max_features': ['auto', 'sqrt', 'log2', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

param_dist5 = {
    'C': np.arange(1, 20, 0.1),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': np.arange(0.1, 10, 0.1),
    'degree': [*range(0, 20, 1)],
}

knn0 = KNeighborsClassifier()
clf0 = tree.DecisionTreeClassifier(random_state=3)
rf0 = RandomForestClassifier(random_state=3)
gbc0 = GradientBoostingClassifier(random_state=3)
svc0 = svm.SVC(random_state=3)


def trainmodel(model, param_dist):
    m = model
    grid = RandomizedSearchCV(m, param_dist, scoring='accuracy', n_iter=20, cv=loovc, n_jobs=-1)
    grid.fit(X_train, Y_train.values.ravel())
    best_estimator = grid.best_estimator_
    print(grid.best_params_)
    print(grid.best_score_)
    m = grid.best_estimator_

    # learning curve
    # train_sizes, train_scores, test_scores = learning_curve(m, X_train, Y_train.values.ravel(), cv=loovc, n_jobs=-1,
    #                                                         scoring='accuracy', shuffle=True)
    # # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # plt.ylim((0.1, 1.1))
    # plt.xlabel("training examples")
    # plt.ylabel("score")
    # plt.grid()
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
    #                 color='r')
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
    #                 color='g')
    #
    # plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='train score')
    # plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='validation score')
    # plt.legend(loc='best')
    # plt.show()

    # validation_curve
    train_scores, test_scores = validation_curve(m, X_train, Y_train.values.ravel(), param_name='p',
                                                 param_range=[i for i in range(1, 6)],
                                                 cv=loovc, scoring='accuracy', n_jobs=-1)
    # plt.ylim((0.1, 1.1))
    plt.xlabel('number of P')
    plt.ylabel('accuracy')
    plt.grid()
    # plt.plot([i for i in range(1, 11)], np.mean(train_scores, axis=1), 'o-', color='r', label='training')
    plt.plot([i for i in range(1, 6)], np.mean(test_scores, axis=1), 'o-', color='g', label='validation')
    plt.legend(loc='best')
    plt.show()
    return m


def performance(m):
    m.fit(X_train, Y_train.values.ravel())

    # 混淆矩阵
    print(m)
    predictions = m.predict(X_test)
    print("accuracy")
    print(accuracy_score(Y_test, predictions))
    print("recall")
    print(recall_score(Y_test, predictions, average=average))
    print("precision:")
    print(precision_score(Y_test, predictions, average=average))
    print("f1")
    print(f1_score(Y_test, predictions, average=average))
    print("roc_auc")
    print(roc_auc_score(Y_test, predictions, multi_class=multi_class))
    # print(predictions)
    cm = confusion_matrix(Y_test.values.ravel(), predictions, labels=labels)
    # print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.xlabel(m)
    plt.show()
    # classification_report包括f1，recall，precision
    print(classification_report(Y_test.values.ravel(), predictions, target_names=target_names))


# For self only
# knn = KNeighborsClassifier(n_neighbors=10, p=1, weights='distance', leaf_size=27, algorithm='kd_tree')
#
# clf = tree.DecisionTreeClassifier(splitter='best', min_samples_split=14, min_samples_leaf=6,
#                                   max_features=6, max_depth=8, criterion='gini', random_state=3,
#                                   class_weight={0: 1, 1: 1})
#
# rf = RandomForestClassifier(random_state=3, criterion='gini', n_estimators=120, max_depth=5,
#                             min_samples_split=16, min_samples_leaf=11, max_features=5,
#                             class_weight={0: 0.6, 1: 0.1})
#
# gbc = GradientBoostingClassifier(random_state=3, learning_rate=0.1, n_estimators=50, max_depth=5,
#                                  min_samples_split=4, min_samples_leaf=3, max_features='sqrt',
#                                  subsample=0.8)
#
# svc = svm.SVC(random_state=3, kernel='linear', gamma=3.7, degree=19, C=2.5)

# For both stacking and self
knn2 = KNeighborsClassifier(n_neighbors=8, p=3, weights='distance', leaf_size=25, algorithm='brute')

clf2 = tree.DecisionTreeClassifier(splitter='best', min_samples_split=19, min_samples_leaf=3,
                                   max_features=4, max_depth=7, criterion='gini', random_state=3)

rf2 = RandomForestClassifier(random_state=3, criterion='entropy', n_estimators=50, max_depth=9,
                             min_samples_split=19, min_samples_leaf=9, max_features='auto')

gbc2 = GradientBoostingClassifier(random_state=3, learning_rate=0.1, n_estimators=1000, max_depth=4,
                                  min_samples_split=12, min_samples_leaf=12, max_features=4,
                                  subsample=0.6)

svc2 = svm.SVC(random_state=3, kernel='ebf', gamma=0.7, degree=3, C=3.4)

knn1 = trainmodel(knn0, param_dist1)
# clf1 = trainmodel(clf0, param_dist2)
# rf1 = trainmodel(rf0, param_dist3)
# gbc1 = trainmodel(gbc0, param_dist4)
# svc1 = trainmodel(svc0, param_dist5)


# performance(knn1)
# performance(clf1)
# performance(rf1)
# performance(gbc1)
# performance(svc1)


# visualize optimal combinations
# k_range = range(1, 6)
# cv_scores = []
# for n in k_range:
#     knn = KNeighborsClassifier(n_neighbors=7, p=n, weights='distance', leaf_size=32, algorithm='auto')
#     scores = cross_val_score(knn, X_train, Y_train.values.ravel(), cv=loovc, scoring='accuracy')
#     cv_scores.append(scores.mean())
# plt.plot(k_range, cv_scores)
# plt.xlabel('number of P')
# plt.ylabel('Accuracy')
# plt.show()


