#!/usr/bin/python3
# -*- coding: utf-8 -*-


class HopfildNetwork:
    """
    НС Хопфилда
    """

    def __init__(self, f, len_vector):
        """
        Конструктор класса
        :param len_vector: Размер вектора входных "изображений"
        :param f: Функция активации
        """
        self.len_vector = len_vector
        self.f = f

        # Матрица весов
        self.weight = [[0 for x in range(len_vector)] for y in range(len_vector)]

    def remember(self, image1, image2, image3):
        """
        Запоминание образов.
        Формируется матрица весов по заданным образам для запоминания.
        """
        for j in range(self.len_vector):
            for k in range(self.len_vector):
                self.weight[j][k] = image1[j] * image1[k] + image2[j] * image2[k] + image3[j] * image3[k]
                if j == k:
                    self.weight[j][k] = 0
        print(self.weight)

    def recognize(self, input_signal):
        """
        Распознавание
        :param input_signal: входной вектор
        """
        y = [[0 for x in range(self.len_vector)] for y in range(self.len_vector)]

        while y != input_signal:
            y = input_signal

            for k in range(self.len_vector):
                # Считаем комбинированный вход в асинхронном режиме
                net = sum([self.weight[j][k] * input_signal[j] for j in range(k - 1)]) + \
                      sum([self.weight[j][k] * y[j] for j in range(k + 1, self.len_vector)])

                input_signal[k] = self.f(net, y[k])

        return input_signal