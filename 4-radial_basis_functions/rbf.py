#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class RBF:
    """
    Класс для НС с радиальными базисными функциями для булевой функции с 4-мя переменными
    """

    def __init__(self, f, x, centers, y_func, learn_k):
        """
        Конструктор класса
        :param centers: центры RBF
        :param y_func: функция активации выходного нейрона
        :param j_func: функция активации скрытого нейрона
        :param teacher: обучающая выборка
        """
        self.centers = centers
        self.y_func = y_func
        self.J = len(centers)
        self.learn_k = learn_k
        self.f = f
        self.x = x

        self.epochs = 0
        self.weights = [0 for i in range(self.J + 1)]
        self.errors = []

    def j_net(self, x, center):
        """
        Гауссова ФА
        :param center: j-й центр RBF
        :param x: входное значение (a, b, c, d)
        """
        return math.exp(-sum([(x[i] - center[i]) ** 2 for i in range(len(x))]))

    def net(self, input_j):
        """
        Комбинированный вход на y
        :param input_j:
        :param w0: смещение
        """
        return sum(self.weights[j] * input_j[j]
                   for j in range(self.J)) + self.weights[3]

    def learning(self):
        """
        Процесс обучения
        """

        error = 16

        while error > 0:
            y = []
            for l in range(len(self.x)):
                # Элементарный шаг обучения!

                # Вычисление прогнозируемого значения
                j_output = [self.j_net(self.x[l], self.centers[j]) for j in range(self.J)]
                output = self.y_func(self.net(j_output))
                y.append(output)

                # Разница между целевым и реальным выходом
                delta = self.f[l] - output

                bias = self.weights[3]
                self.weights = [self.weights[j] + self.learn_k * delta * j_output[j] for j in range(self.J)]
                self.weights.append(bias + self.learn_k * delta * 1)

            # Вычисление ошибки
            error = sum([y[i] ^ self.f[i] for i in range(len(self.f))])
            self.errors.append(error)
            print("Ошибка({}) = {}, выход = {}".format(self.epochs, error, y))
            self.epochs += 1

            if self.epochs > 100:
                print("Обучить не удалось!")
                break

        return self.epochs, self.errors
