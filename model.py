import random
import randomcolor

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier


class DataGenerator:
    def __init__(self):
        pass

    @staticmethod
    def divided_square(size, gaussian_noise):
        x = np.random.rand(size, 2)
        y = x[:, 1] > x[:, 0]
        if len(x) != len(y):
            raise Exception("Incorrect vectors length")
        noise = gaussian_noise * np.random.normal(0, 1, (size, 2))
        x += noise
        return x, y

    @staticmethod
    def chessboard(size, squares, gaussian_noise):
        x = np.random.rand(size, 2)
        y = []
        for i in range(len(x)):
            for col, row in product(range(squares), range(squares)):
                if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
                    if (row <= x[i, 0] * squares < row + 1) and (col <= x[i, 1] * squares < col + 1):
                        y.append(True)
                        break
                else:
                    if (row <= x[i, 0] * squares < row + 1) and (col <= x[i, 1] * squares < col + 1):
                        y.append(False)
                        break
        if len(x) != len(y):
            raise Exception("Incorrect vectors length")
        noise = gaussian_noise * np.random.normal(0, 1, (size, 2))
        x += noise
        return x, np.asarray(y)


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def split_data(x, y, ratio):
        combined = list(zip(x, y))
        random.shuffle(combined)
        x, y = zip(*combined)
        x = np.array(x)
        y = np.array(y)
        teach_size = int(len(x) * ratio)
        return x[:teach_size], x[teach_size:], y[:teach_size], y[teach_size:]

    @staticmethod
    def count_error(result, predict_result):
        if len(result) != len(predict_result):
            raise Exception("Arguments length error!")

        correct = 0
        for i in range(0, len(result)):
            if result[i] == predict_result[i]:
                correct += 1
        accuracy = correct / len(result)
        error = 1 - accuracy
        return error, accuracy


class PlotGenerator:
    @staticmethod
    def show_generated_data(x, y):
        plt.scatter(x[y == 0, 0], x[y == 0, 1], c="b", marker='.')
        plt.scatter(x[y == 1, 0], x[y == 1, 1], c="g", marker='x')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    @staticmethod
    def show_accuracy_chart(efficiency_array):
        rand_color = randomcolor.RandomColor()
        colors = rand_color.generate(count=len(efficiency_array))
        for i in range(0, len(efficiency_array)):
            plt.plot(np.array(efficiency_array[i])[:, 0],
                     np.array(efficiency_array[i])[:, 1],
                     colors[i],
                     label="Series %d" % (i + 1))
        plt.legend(loc='best')
        plt.xlabel('k')
        plt.ylabel('efficiency')
        plt.show()

    @staticmethod
    def show_decision_boundaries(x, y, k_opt):
        h = .02
        color_map_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        color_map_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        for weights in ['uniform', 'distance']:
            clf = KNeighborsClassifier(k_opt, weights=weights)
            clf.fit(x, y)

            x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
            y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=color_map_light)

            plt.scatter(x[:, 0], x[:, 1], c=y, cmap=color_map_bold,
                        edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("K = %i, Weights = '%s'"
                      % (k_opt, weights))
        plt.show()
