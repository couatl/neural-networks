#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math


class Perceptron:
    def __init__(self, learn_k, X, y, p, epoch=100):
        """
        Конструктор класса
        :param learn_k: норма
        :param y_true: реальные значения ряда
        :param size_window: размер прогнозируемого окна
        """
        self.X = X
        self.y = y
        self.learn_k = learn_k
        self.p = p
        self.epoch = epoch

        self.epochs = 0
        self.weights = [0 for i in range(p + 1)]
        self.weights[0] = 1

    def net(self, weights, x):
        return weights[0] + sum(weights[k + 1] * x[k] for k in range(self.p))

    def error(self, y, y_true):
        """
        Функция суммарной среднеквадратичной ошибки
        :param y: спрогнозированные значения
        :param y_true: реальные значения y(t[i])
        :return: Суммарная среднеквадратичная ошибка
        """
        return math.sqrt(sum((y[i] - y_true[i]) ** 2 for i in range(len(y))))

    def learning(self):
        """
        Процесс обучения
        """

        y_forecast = []
        self.errors = []
        e = 1
        k = 0
        while k < self.epoch:
            y_forecast = []

            for i in range(len(self.X)):
                output = self.net(self.weights, self.X[i])
                y_forecast.append(output)

                delta = self.y[i + self.p] - output
                #         print("Output = {}, y={}".format(output, y[i + p]))

                for j in range(len(self.weights) - 1):
                    self.weights[j + 1] = self.weights[j + 1] + 0.3 * delta * self.X[i][j]
                self.weights[0] += 0.3 * delta
                #         print("Weights: {}".format(weights))

            e = self.error(y_forecast, self.y[4:])
            self.errors.append(e)
            print("Output({}) = {}, error = {}".format(k, y_forecast[:3], e))
            k += 1
        return y_forecast, self.errors, k

        # for i in range(self.size_window, len(self.y_true) - 1):
        #     # Прогнозируемое значение ряда в момент времени n для авторегрессионной модели
        #     y_calculate = self.net(self.weight, self.y_true, self.size_window, i)
        #
        #     # Ошибка прогноза
        #     delta = self.y_true[i] - y_calculate
        #
        #     # Коррекция весов по правилу Видроу-Хоффа
        #     self.weight = sum(delta * self.learn_k * self.y_true[i + j - self.size_window]
        #                       for j in range(self.size_window))
        #     print("Вычисление на " + self.weight + "эпохе")