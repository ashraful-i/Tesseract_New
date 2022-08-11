# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from math import sqrt
import pandas as pd
import sys
import matplotlib.pyplot as plt


def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2  # (x1-x2)**2+(y1-y2)**2
    return sqrt(distance)


def get_row_val_hr(test_rows):
    # print(test_rows)
    df1 = pd.DataFrame(columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                                'smoking', 'time', 'DEATH_EVENT'])
    # print(df1)
    age = 0
    anaemia = 0
    creatinine_phosphokinase = 0
    diabetes = 0
    ejection_fraction = 0
    high_blood_pressure = 0
    platelets = 0
    serum_creatinine = 0
    serum_sodium = 0
    sex = 0
    smoking = 0
    time = 0
    death_event = 0
    for cnt, items in enumerate(test_rows):
        # print(cnt)
        for item in items:
            age += item.age
            anaemia += item.anaemia
            creatinine_phosphokinase += item.creatinine_phosphokinase
            diabetes += item.diabetes
            ejection_fraction += item.ejection_fraction
            high_blood_pressure += item.high_blood_pressure
            platelets += item.platelets
            serum_creatinine += item.serum_creatinine
            serum_sodium += item.serum_sodium
            sex += item.sex
            smoking += item.smoking
            time += item.time
            death_event += item.DEATH_EVENT

        age = age / len(items)
        anaemia = anaemia / len(items)
        creatinine_phosphokinase = creatinine_phosphokinase / len(items)
        diabetes = diabetes / len(items)
        ejection_fraction = ejection_fraction / len(items)
        high_blood_pressure = high_blood_pressure / len(items)
        platelets = platelets / len(items)
        serum_creatinine = serum_creatinine / len(items)
        serum_sodium = serum_sodium / len(items)
        sex = sex / len(items)
        smoking = smoking / len(items)
        time = time / len(items)
        death_event = death_event / len(items)
        datafrm = {'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase,
                   'diabetes': diabetes, 'ejection_fraction': ejection_fraction,
                   'high_blood_pressure': high_blood_pressure,
                   'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
                   'sex': sex,
                   'smoking': smoking, 'time': time, 'DEATH_EVENT': death_event}
        df1 = pd.concat([df1, pd.DataFrame.from_records([datafrm])], ignore_index=True)
    # print(df1)
    return df1


def get_minimum_point(dataset, n, train_set):
    distance = [[-1 for x in range(n)] for y in range(len(dataset)+1)]
    min_dist = sys.maxsize
    min_idx = -1
    for cnt_dataset, i in dataset.iterrows():

        for cnt_trainset, j in train_set.iterrows():
            tr_lst = [j.age, j.anaemia, j.creatinine_phosphokinase, j.diabetes, j.ejection_fraction,
                      j.high_blood_pressure, j.platelets, j.serum_creatinine, j.serum_sodium, j.sex, j.smoking, j.time,
                      j.DEATH_EVENT]
            df_lst = [i.age, i.anaemia, i.creatinine_phosphokinase, i.diabetes, i.ejection_fraction,
                      i.high_blood_pressure, i.platelets, i.serum_creatinine, i.serum_sodium, i.sex, i.smoking, i.time,
                      i.DEATH_EVENT]

            dis_temp = Euclidean_distance(tr_lst, df_lst)
            if dis_temp < min_dist:
                min_dist = dis_temp
                min_idx = cnt_trainset
        #print(cnt_dataset, min_idx)
        distance[cnt_dataset][min_idx] = i
    #(distance)
    test_rows = [[] for x in range(n)]
    for cnt_row, row in enumerate(distance):
        for cnt, item in enumerate(row):
            if type(item) == int:
                continue
            # print(cnt, item)
            #distance[cnt_row][cnt] = item
            test_rows[cnt].append(item)
    #print(test_rows)
    given_points = get_row_val_hr(test_rows)
    global test_r_1
    test_r_1 = given_points
    '''
    #print(given_points)
    for cnt_g_p, item in given_points.iterrows():
        old_set = train_set.iloc[[cnt_g_p]]
        #print(old_set)
        new_val = item.age + item.anaemia + item.creatinine_phosphokinase + item.diabetes + item.ejection_fraction
        +item.high_blood_pressure + item.platelets + item.serum_creatinine + item.serum_sodium + item.sex + item.smoking
        + item.time
        #print(new_val)
        old_val = old_set.age.values[0] + old_set.anaemia.values[0] + old_set.creatinine_phosphokinase.values[0]
        + old_set.diabetes.values[0] + old_set.ejection_fraction.values[0] + old_set.high_blood_pressure.values[0]
        + old_set.platelets.values[0] + old_set.serum_creatinine.values[0] + old_set.serum_sodium.values[0]
        + old_set.sex.values[0] + old_set.smoking.values[0] + old_set.time.values[0]
        diff = abs(new_val - old_val)
        print(diff)
        if (diff < 10):
            global test_r_1
            test_r_1 = given_points
            return
    GetTestAreas(dataset, n, given_points)
    '''
def GetTestAreas(dataset, n, train_set):
    distance = [[-1 for x in range(n)] for y in range(len(dataset)+1)]
    global min_point
    for cnt_dataset, i in dataset.iterrows():
        min_dist = sys.maxsize
        min_idx = -1
        for cnt_trainset, j in train_set.iterrows():
            tr_lst = [j.age, j.anaemia, j.creatinine_phosphokinase, j.diabetes, j.ejection_fraction,
                      j.high_blood_pressure, j.platelets, j.serum_creatinine, j.serum_sodium, j.sex, j.smoking, j.time,
                      j.DEATH_EVENT]
            df_lst = [i.age, i.anaemia, i.creatinine_phosphokinase, i.diabetes, i.ejection_fraction,
                      i.high_blood_pressure, i.platelets, i.serum_creatinine, i.serum_sodium, i.sex, i.smoking, i.time,
                      i.DEATH_EVENT]

            dis_temp = Euclidean_distance(tr_lst, df_lst)
            if dis_temp < min_dist:
                min_dist = dis_temp
                min_idx = cnt_trainset
                min_point = i

    test_rows = [[] for x in range(n)]

    for cnt_row, row in enumerate(distance):
        for cnt, item in enumerate(row):
            if type(item) == int:
                continue
            # print(cnt, item)
            distance[cnt_row][cnt] = item
            test_rows[cnt].append(item)
    given_points = get_row_val_hr(test_rows)

    for cnt_g_p, item in given_points.iterrows():
        old_set = train_set.iloc[[cnt_g_p]]
        new_val = item.age + item.anaemia + item.creatinine_phosphokinase + item.diabetes + item.ejection_fraction
        +item.high_blood_pressure + item.platelets + item.serum_creatinine + item.serum_sodium + item.sex + item.smoking
        + item.time

        old_val = old_set.age.values[0] + old_set.anaemia.values[0] + old_set.creatinine_phosphokinase.values[0]
        + old_set.diabetes.values[0] + old_set.ejection_fraction.values[0] + old_set.high_blood_pressure.values[0]
        + old_set.platelets.values[0] + old_set.serum_creatinine.values[0] + old_set.serum_sodium.values[0]
        + old_set.sex.values[0] + old_set.smoking.values[0] + old_set.time.values[0]
        diff = abs(new_val - old_val)
        print(diff)
        if (diff < 10):
            global test_r_1
            test_r_1 = given_points
            return
    GetTestAreas(dataset, n, given_points)
def divide_dataset(testing_data, train_set, n):
    distance = [[-1 for x in range(num_of_set)] for y in range(len(testing_data))]
    for cnt_dataset, i in testing_data.iterrows():
        min_dist = sys.maxsize
        min_idx = -1
        for cnt_trainset, j in train_set.iterrows():
            tr_lst = [j.age, j.anaemia, j.creatinine_phosphokinase, j.diabetes, j.ejection_fraction,
                      j.high_blood_pressure, j.platelets, j.serum_creatinine, j.serum_sodium, j.sex, j.smoking, j.time]
            df_lst = [i.age, i.anaemia, i.creatinine_phosphokinase, i.diabetes, i.ejection_fraction,
                      i.high_blood_pressure, i.platelets, i.serum_creatinine, i.serum_sodium, i.sex, i.smoking, i.time]
            dis_temp = Euclidean_distance(tr_lst, df_lst)
            if dis_temp < min_dist:
                min_dist = dis_temp
                min_idx = cnt_trainset
        distance[cnt_dataset][min_idx] = i
    test_rows = [[] for x in range(n)]

    for cnt_row, row in enumerate(distance):
        for cnt, item in enumerate(row):
            if type(item) == int:
                continue
            # print(cnt, item)
            distance[cnt_row][cnt] = item
            test_rows[cnt].append(item)
    df1 = pd.DataFrame(columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                                'smoking', 'time', 'DEATH_EVENT'])
    df_list = []
    for x in range(n):
        df_list.append(df1)

    for item_cnt, item in enumerate(test_rows):
        for ii in item:
            df_list[item_cnt] = pd.concat([df_list[item_cnt], pd.DataFrame.from_records([ii])], ignore_index=True)
    #print(df_list)
    for df_s in df_list:
        exclude = ['']
        df_s.loc[:, df_s.columns.difference(exclude)].hist()

        # df_s.plot(kind="bar", stacked=True)
        plt.show()
    pass

def get_accuracy(testing_data, death_set, not_death_set):
    pass

if __name__ == '__main__':
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',', header=[0])
    df = df.sample(n=df.shape[0], random_state=10)

    mask = int(df.shape[0] * 80 / 100)
    training_data = df[:mask]
    testing_data = df[mask:]

    num_of_set = 2
    training_data = training_data.reset_index(drop=True)
    testing_data = testing_data.reset_index(drop=True)
    #print(training_data)
    death_train_set = training_data.loc[training_data['DEATH_EVENT'] == 1]
    death_train_set = death_train_set.reset_index(drop=True)
    #print(death_train_set.head(1))

    global min_point
    min_point = -1
    #print(death_train_set.head(1))
    death_train_full_set = death_train_set.iloc[1: , :]
    death_train_sample = death_train_set.head(1)
    #print(death_train_set.iloc[1: , :])
    global test_r_1

    get_minimum_point(death_train_full_set, 1, death_train_sample)
    death_set = test_r_1
    #print(test_r_1.columns.tolist())
    print("Dead")
    for col in death_set.columns.tolist():
        print(death_set[col])
    #global test_r_1
    not_death_train_set = training_data.loc[training_data['DEATH_EVENT'] == 0]
    not_death_train_set = not_death_train_set.reset_index(drop=True)
    print(not_death_train_set)
    not_death_train_full_set = not_death_train_set.iloc[1:, :]
    not_death_train_sample = not_death_train_set.head(1)
    get_minimum_point(not_death_train_full_set, 1, not_death_train_sample)
    not_death_set = test_r_1

    print("Survived")

    for col in not_death_set.columns.tolist():
        print(not_death_set[col])


    #get_minimum_point(death_train_set.iloc[1: , :], 1, death_train_set.head(1))

    '''
    train_set = training_data.sample(n=num_of_set, random_state=10).reset_index(drop=True)
    print(train_set)
    training_data = training_data.reset_index(drop=True)
    testing_data = testing_data.reset_index(drop=True)
    df = df.reset_index(drop=True)
    global test_r_1
    GetTestAreas(training_data, num_of_set, train_set)
    print(test_r_1)
    '''
    get_accuracy(testing_data, death_set, not_death_set)
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
