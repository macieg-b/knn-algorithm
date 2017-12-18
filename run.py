from sklearn.neighbors import KNeighborsClassifier

from model import DataGenerator, PlotGenerator, DataManager

N = 5000
REPETITION = 1
STEP = 50

x_1, y_1 = DataGenerator.divided_square(N, 0.04)
x_2, y_2 = DataGenerator.chessboard(N, 2, 0.04)
x_3, y_3 = DataGenerator.chessboard(N, 3, 0.04)
x_4, y_4 = DataGenerator.chessboard(N, 4, 0.04)
x_5, y_5 = DataGenerator.chessboard(N, 5, 0.04)

data_to_test = list()
data_to_test.append((x_1, y_1))
data_to_test.append((x_2, y_2))
data_to_test.append((x_3, y_3))
data_to_test.append((x_4, y_4))
data_to_test.append((x_5, y_5))

efficiency_list = []
for i in range(len(data_to_test)):
    x = data_to_test[i][0]
    y = data_to_test[i][1]
    PlotGenerator.show_generated_data(x, y)
    teach_data, test_data, teach_result, test_result = DataManager.split_data(x, y, 0.5)

    vec = range(1, len(teach_data), STEP)
    efficiency = []
    for k in vec:
        neighbour_efficiency = 0
        print("Iteration %d" % k)
        for i in range(REPETITION):
            kNeighborsClassifier = KNeighborsClassifier(k)
            kNeighborsClassifier.fit(teach_data, teach_result)
            neighbour_efficiency += kNeighborsClassifier.score(test_data, test_result)
        neighbour_efficiency /= REPETITION
        efficiency.append((k, neighbour_efficiency))

    efficiency_list.append(efficiency)

PlotGenerator.show_accuracy_chart(efficiency_list)
k_opt = [2280, 1300, 503, 307, 160]

for i in range(len(k_opt)):
    x = data_to_test[i][0]
    y = data_to_test[i][1]
    PlotGenerator.show_decision_boundaries(x, y, k_opt[i])
