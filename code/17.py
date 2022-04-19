import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

abvenues = [
    'TRENDS ECOL EVOL',
    'NAT ECOL EVOL',
    'ANNU REV ECOL EVOL S',
    'FRONT ECOL ENVIRON',
    'GLOBAL CHANGE BIOL',
    'ECOL MONOGR',
    'ISME J'
]
    
i = 42
years = ['2019', '2018', '2017']

df_training_set = pd.read_csv('aatest2.csv')

for ve in abvenues:
    X = pd.read_csv('D:/study/毕设/journal/test2/' + str(i) + '1.csv')
    XX = pd.read_csv('D:/study/毕设/journal/test2/' + str(i) + '3.csv')
    XXX = pd.read_csv('D:/study/毕设/journal/test2/' + str(i) + '2.csv')
    # print(X)

    # 前三年总被引量 年份要改
    df2 = X[X['Citing Journal'].str.contains('ALL Journals')][years]
    three_years_total = df2.values.tolist()
    two_years_total_cited_count = three_years_total[0][0] + three_years_total[0][1]
    three_years_total_cited_count = three_years_total[0][0] + three_years_total[0][1] + three_years_total[0][2]

    # 前三年总引用量 年份要改
    three_years_citing_total = XXX[XXX['Cited Journal'].str.contains('ALL Journals')][years].values.tolist()
    two_years_total_citing_count = three_years_citing_total[0][0] + three_years_citing_total[0][1]
    three_years_total_citing_count = three_years_citing_total[0][0] + three_years_citing_total[0][1] + \
                                     three_years_citing_total[0][2]

    # # 判断cited文件里有没有目标期刊
    df5 = X[X['Citing Journal'] == ve]
    if df5.empty:
        # feature1
        total_self_count = 0

        # feature2:two_years_self_citing_rate & two_years_self_citing_count
        two_years_self_count = 0

        # feature3:two_years_self_citing_rate
        three_years_self_count = 0
    else:
        # feature1
        total_self_count = df5['All Yrs'].values[0]

        # feature2:two_years_self_citing_rate & two_years_self_citing_count 年份要改
        df = df5[years]
        three_years_self = df.values.tolist()
        two_years_self_count = three_years_self[0][0] + three_years_self[0][1]

        # feature3:two_years_self_citing_rate
        three_years_self_count = three_years_self[0][0] + three_years_self[0][1] + three_years_self[0][2]

    # feature1:total_self_citing_rate
    df4 = X.loc[[0, 1]]
    total_cited_count = df4['All Yrs'].values.tolist()
    # print(total_cited_count)
    df_training_set.loc[i, 'total_self_citing_rate'] = round(int(total_self_count) / int(total_cited_count[0]), 4)

    # feature2:two_years_self_citing_rate & two_years_self_citing_count
    print(two_years_self_count)
    df_training_set.loc[i, 'before_two_years_self_citing_count'] = two_years_self_count
    print(round(two_years_self_count / two_years_total_cited_count, 4))
    df_training_set.loc[i, 'two_years_self_citing_rate'] = round(
        two_years_self_count / two_years_total_cited_count,
        4)

    # feature3:two_years_self_citing_rate
    print(round(three_years_self_count / three_years_total_cited_count, 4))
    df_training_set.loc[i, 'three_years_self_citing_rate'] = round(
        three_years_self_count / three_years_total_cited_count, 4)

    # feature7: paper_count_in_given_year 年份要改
    XX2 = XX[XX.index == 2020]
    XX2 = XX2.dropna(axis=0, subset=["Immediacy Index"])
    # print(XX)
    df7 = XX2['Immediacy Index'].values.tolist()
    print(df7[0])
    df_training_set.loc[i, 'paper_count_in_given_year'] = df7[0]

    # feature4: average_pub_rate 年份要改
    XX = XX[XX.index <= 2019]
    XX = XX.dropna(axis=0, subset=["Immediacy Index"])
    # print(XX)
    df3 = XX['Immediacy Index']
    total = 0
    paper_count_total = df3.values.tolist()
    for ele in range(0, len(paper_count_total)):
        if isinstance(paper_count_total[ele], str):
            paper_count_total[ele] = int(paper_count_total[ele])
    print(paper_count_total)
    for ele in range(0, len(paper_count_total)):
        total = total + paper_count_total[ele]
    print(round(total / len(paper_count_total), 4))
    df_training_set.loc[i, 'average_pub_rate'] = round(total / len(paper_count_total), 4)

    # feature5: outgoing reference
    X = X.drop(index=[0, 1], axis=0)
    journal_citing_count_total = X.shape[0] + total_cited_count[1]
    print(round(total_cited_count[0] / journal_citing_count_total, 4))
    df_training_set.loc[i, 'outgoing_references'] = round(total_cited_count[0] / journal_citing_count_total, 4)

    # feature6: incoming citations
    # print(XXX)
    df6 = XXX.loc[[0, 1]]
    total_citing_count = df6['All Yrs'].values.tolist()
    XXX = XXX.drop(index=[0, 1], axis=0)
    journal_cited_count_total = XXX.shape[0] + total_citing_count[1]
    print(round(total_citing_count[0] / journal_cited_count_total, 4))
    df_training_set.loc[i, 'incoming_citations'] = round(total_citing_count[0] / journal_cited_count_total, 4)

    # feature9:total_other_citing_rate
    print((total_citing_count[0] - total_self_count) / total_citing_count[0])
    df_training_set.loc[i, 'total_other_cited_rate'] = round(
        (total_citing_count[0] - total_self_count) / total_citing_count[0], 4)

    # feature10:two_years_other_cited_rate
    print((two_years_total_citing_count - two_years_self_count) / two_years_total_citing_count)
    df_training_set.loc[i, 'two_years_other_cited_rate'] = round(
        (two_years_total_citing_count - two_years_self_count) / two_years_total_citing_count, 4)

    # feature11:three_years_other_cited_rate
    print((three_years_total_citing_count - three_years_self_count) / three_years_total_citing_count)
    df_training_set.loc[i, 'three_years_other_cited_rate'] = round(
        (three_years_total_citing_count - three_years_self_count) / three_years_total_citing_count, 4)

    print(i)
    i = i + 1

print(i)
print(df_training_set)
df_training_set.to_csv('aatest2.csv', encoding='utf-8', index=False,
                       columns=['abnormal', 'venue_name', 'total_self_citing_rate',
                                'before_two_years_self_citing_count',
                                'paper_count_in_given_year', 'Reference_count_of_each_paper_published',
                                'two_years_self_citing_rate',
                                'three_years_self_citing_rate', 'year', 'average_pub_rate', 'incoming_citations',
                                'outgoing_references', 'stacking', 'total_other_cited_rate',
                                'two_years_other_cited_rate', 'three_years_other_cited_rate'
                                ])
