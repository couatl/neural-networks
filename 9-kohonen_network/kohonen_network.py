#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


class KohonenNetwork:
    """
    НС Кохонена
    """

    def __init__(self, values, centers):
        """
        Конструктор класса
        :param values: массив из (x, y) значений для кластеризации
        :param centers: массив из (x, y) центров кластеров
        """
        self.values = np.array(values)
        self.centers = np.array(centers)

        # Матрица весов
        self.weights = np.zeros((len(values), len(centers)))

    def euclidean_distance(self, a, b):
        """
        Функция расчета Евклидова расстояния между матрицами а и b
        """
        return np.linalg.norm(a - b)

    def calculate_weights(self):
        """
        Вычисление матрицы весов - принадлежности к кластерам
        """

        # Считаем расстояние для каждого нейрона и входного сигнала
        for value_i in range(len(self.values)):
            for center_i in range(len(self.centers)):
                self.weights[value_i][center_i] = self.euclidean_distance(self.values[value_i], self.centers[center_i])

        # Выполнение правила сильнейшего: минимальному элементу присваиваем 1, остальным 0
        for value_i in range(len(self.values)):
            min_index = self.weights[value_i].argmin()
            self.weights[value_i][min_index] = 1
            self.weights[value_i][0:min_index] = 0
            self.weights[value_i][min_index + 1:] = 0

        return self.weights
