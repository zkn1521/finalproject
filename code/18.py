import pandas as pd

# pd.set_option('display.max_columns', None)

abvenues = ['TRENDS ECOL EVOL',
            'NAT ECOL EVOL',
            'ANNU REV ECOL EVOL S',
            'FRONT ECOL ENVIRON',
            'GLOBAL CHANGE BIOL',
            'ECOL MONOGR',
            'ISME J'
            ]

i = 42


# 删除文件前几行
def delete_first_lines(filename, count):
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[count:])
    fout.write(b)


# 删除文件后几行
def delete_last_lines(filename, count):
    file = open(filename)
    lines = file.readlines()
    lines = lines[:-count]
    file.close()
    w = open(filename, 'w')
    w.writelines(lines)


for ve in abvenues:
    delete_first_lines('D:/study/毕设/journal/test2/' + str(i) + '3.csv', 5)
    delete_last_lines('D:/study/毕设/journal/test2/' + str(i) + '3.csv', 4)
    delete_first_lines('D:/study/毕设/journal/test2/' + str(i) + '1.csv', 5)
    delete_first_lines('D:/study/毕设/journal/test2/' + str(i) + '2.csv', 5)
    delete_last_lines('D:/study/毕设/journal/test2/' + str(i) + '1.csv', 3)
    delete_last_lines('D:/study/毕设/journal/test2/' + str(i) + '2.csv', 3)
    i = i + 1

print(i)
