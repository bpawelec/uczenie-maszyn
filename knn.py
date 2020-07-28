import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from common.import_data import ImportData
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

if __name__ == "__main__":
    data_set = ImportData()

    y: np.ndarray = data_set.import_classes(['class']).ravel()

    x_tmp: np.ndarray = data_set.import_columns('./dataset3/pokerhand.csv', ['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])

    kf_for_test = KFold(n_splits=10, shuffle=True)
    result_tmp = next(kf_for_test.split(x_tmp), None)
    x_test = x_tmp[result_tmp[1]]
    y_test = y[result_tmp[1]]


    kf_AllKNN = KFold(n_splits=5, shuffle=True)
    #x = data_set.import_columns(['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    x_AllKNN: np.ndarray = data_set.import_columns('./dataset3/reduced_dataset_AllKNN_pokerhand.csv',['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    #x = data_set.import_all_data()
    y = data_set.import_classes_for_reduced_datasets('./dataset3/reduced_dataset_AllKNN_pokerhand.csv',np.array(['class']))

    result_AllKNN = next(kf_AllKNN.split(x_AllKNN), None)
    x_train_AllKNN = x_AllKNN[result_AllKNN[0]]
    #x_test = x[result[1]]
    y_train_AllKNN = y[result_AllKNN[0]]
    #y_test = y[result_AllKNN[1]]
    NN_AllKNN = KNeighborsClassifier(n_neighbors=5)
    NN_AllKNN.fit(x_train_AllKNN,y_train_AllKNN.ravel())
    predictions_AllKNN = NN_AllKNN.predict(x_test)
    print(NN_AllKNN.score(x_test, y_test))

    f1 = open("reduced_dataset_AllKNN_pokerhand.csv", "w+")
    a_str_AllKNN = '\n'.join(str(x) for x in predictions_AllKNN)
    f1.write(a_str_AllKNN)
    f1.close()


    kf_enn = KFold(n_splits=5, shuffle=True)
    # x = data_set.import_columns(['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    x_enn: np.ndarray = data_set.import_columns('./dataset3/reduced_dataset_enn_pokerhand.csv', ['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    # x = data_set.import_all_data()
    y = data_set.import_classes_for_reduced_datasets('./dataset3/reduced_dataset_enn_pokerhand.csv', np.array(['class']))

    result_enn = next(kf_enn.split(x_enn), None)
    x_train_enn = x_enn[result_enn[0]]
    # x_test = x[result[1]]
    y_train_enn = y[result_enn[0]]
    # y_test = y[result_AllKNN[1]]
    NN_enn = KNeighborsClassifier(n_neighbors=5)
    NN_enn.fit(x_train_enn, y_train_enn.ravel())
    predictions_enn = NN_enn.predict(x_test)
    print(NN_enn.score(x_test, y_test))

    f2 = open("reduced_dataset_enn_pokerhand.csv", "w+")
    a_str_enn = '\n'.join(str(x) for x in predictions_enn)
    f2.write(a_str_enn)
    f2.close()

    kf_ncr = KFold(n_splits=5, shuffle=True)
    # x = data_set.import_columns(['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    x_ncr: np.ndarray = data_set.import_columns('./dataset3/reduced_dataset_ncr_pokerhand.csv', ['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    # x = data_set.import_all_data()
    y = data_set.import_classes_for_reduced_datasets('./dataset3/reduced_dataset_ncr_pokerhand.csv', np.array(['class']))

    result_ncr = next(kf_ncr.split(x_ncr), None)
    x_train_ncr = x_ncr[result_ncr[0]]
    # x_test = x[result[1]]
    y_train_ncr = y[result_ncr[0]]
    # y_test = y[result_AllKNN[1]]
    NN_ncr = KNeighborsClassifier(n_neighbors=5)
    NN_ncr.fit(x_train_ncr, y_train_ncr.ravel())
    predictions_ncr = NN_ncr.predict(x_test)
    print(NN_ncr.score(x_test, y_test))

    f3 = open("reduced_dataset_ncr_pokerhand.csv", "w+")
    a_str_ncr = '\n'.join(str(x) for x in predictions_ncr)
    f3.write(a_str_ncr)
    f3.close()

    kf_original = KFold(n_splits=5, shuffle=True)
    # x = data_set.import_columns(['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    x_original: np.ndarray = data_set.import_columns('./dataset3/pokerhand.csv', ['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    # x = data_set.import_all_data()
    y = data_set.import_classes_for_reduced_datasets('./dataset3/pokerhand.csv', np.array(['class']))

    result_original = next(kf_original.split(x_original), None)
    x_train_original = x_original[result_original[0]]
    # x_test = x[result[1]]
    y_train_original = y[result_original[0]]
    # y_test = y[result_AllKNN[1]]
    NN_original = KNeighborsClassifier(n_neighbors=5)
    NN_original.fit(x_train_original, y_train_original.ravel())
    predictions_original = NN_original.predict(x_test)
    print(NN_original.score(x_test, y_test))

    f4 = open("original_pokerhand.csv", "w+")
    a_str_original = '\n'.join(str(x) for x in predictions_original)
    f4.write(a_str_original)
    f4.close()