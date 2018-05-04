#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math


class Perceptron:
    def __init__(self, learn_k, y_true, size_window):
        """
        Конструктор класса
        :param learn_k: норма
        :param y_true: реальные значения ряда
        :param size_window: размер прогнозируемого окна
        """
        self.y_true = y_true
        self.learn_k = learn_k
        self.size_window = size_window

        self.epochs = 0
        self.weight = [0 for i in range(size_window)]

        self.error_on_last_epochs = 0
        self.vector_ans_on_epoch = []

    def net(self, weight, x, i, size_window, w0=0):
        """
        :param weight: весовые коэффициенты
        :param x: обучающая выборка значений временного ряда
        :param i: момент времени в выборке
        :param size_window: размер окна
        :param w0: вес смещения
        :return: Комбинированный вход нейрона
        """
        return sum(weight[k] * x[i - size_window + k]
                   for k in range(size_window)) + w0

    def forecast_y(self, weight, y_true, size_window):
        """
        :param weight: весовые коэффициенты
        :param y_true: реальные значения
        :param size_window: размер окна данных
        :return: Прогнозируемое значение ряда авторегрессионной модели
        """
        return sum(y_true[j - size_window] * weight[j]
                   for j in range(size_window))

    def learning(self):
        """
        Процесс обучения
        """
        for i in range(self.size_window, len(self.y_true) - 1):
            # Прогнозируемое значение ряда в момент времени n для авторегрессионной модели
            y_calculate = self.net(self.weight, self.y_true, self.size_window, i)

            # Ошибка прогноза
            delta = self.y_true[i] - y_calculate

            # Коррекция весов по правилу Видроу-Хоффа
            self.weight = sum(delta * self.learn_k * self.y_true[i + j - self.size_window]
                              for j in range(self.size_window))

        self.epochs += 1

    def neuron_prediction(self, number):
        """

        :param number:
        :return:
        """
        y_prediction = self.y_true[-self.size_window:]

        for i in range(number):
            next_prediction = sum(y_prediction[j - self.size_window] * self.weight[j]
                                  for j in range(self.size_window))

            y_prediction.append(next_prediction)

        return y_prediction[self.size_window:]

    def error(self, y, y_true):
        """
        Функция суммарной среднеквадратичной ошибки
        :param y: спрогнозированные значения
        :param y_true: реальные значения y(t[i])
        :return: Суммарная среднеквадратичная ошибка
        """
        return math.sqrt(sum((y[i] - y_true[i + self.size_window]) ** 2 for i in range(len(y))))

    def calculate_error(self):
        """
        Функция вычисления суммарной квадратичной ошибки
        """

        # Пересчитывание реального выхода НС
        self.vector_y = [self.net(self.weight, self.y_true, self.size_window)
                         for i in range(self.size_window, len(self.y_true) - 1)]

        return self.error(self.vector_y, self.y_true)
