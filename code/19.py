import pandas as pd
import numpy as np
from sklearn.utils import shuffle

pd.set_option('display.max_columns', None)
# 打乱数据集并分为x，y(label)
df_training = pd.read_csv('aa4.csv')
df_training_shuff = shuffle(df_training, random_state=0)
print(df_training_shuff)
df_training_shuff.to_csv('aa4_x_training_set.csv', encoding='utf-8', index=False,
                         columns=['total_self_citing_rate',
                                  'Reference_count_of_each_paper_published',
                                  'two_years_self_citing_rate',
                                  'three_years_self_citing_rate',
                                  'average_pub_rate',
                                  'incoming_citations',
                                  'outgoing_references',
                                  'total_other_cited_rate',
                                  'two_years_other_cited_rate',
                                  'three_years_other_cited_rate'
                                  ])
df_training_shuff.to_csv('aa4_y_training_set_3_labels.csv', encoding='utf-8', index=False,
                         columns=['abnormal'])
