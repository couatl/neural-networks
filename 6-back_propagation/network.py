#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math


class Network:
    """
    Многослойная Нейронная Сеть с одним скрытым слоем
    """

    def __init__(self, N, J, M, x, t, learn_k, f, df):
        """
        Конструктор класса
        :param N: Размер входного слоя
        :param J: Размер скрытого слоя
        :param M: Размер выходного слоя
        :param learn_k: норма
        :param f: Функция активации
        :param df: Производная функции активации
        :param x: Входной вектор
        :param t: Обучающая выборка
        """
        self.N = N
        self.J = J
        self.M = M
        self.f = f
        self.df = df
        self.x = x
        self.t = t
        self.learn_k = learn_k

        self.epoch = 0
        self.errors = []

        # Первые N значений в первый нейрон следующего слоя, вторые N во второй и т.д.
        self.weight_input = [[0 for x in range(N)] for y in range(J)]
        self.weight_hidden = [[0 for x in range(J)] for y in range(M)]

    def calculate_output(self, x):
        """
        Вычисление выхода МНС по заданному входному сигналу x
        :param x: N входных сигналов // массив длины N
        :return: M выходных сигналов
        """
        f = self.f
        net = self.net
        weight_input = self.weight_input
        weight_hidden = self.weight_hidden

        input_layer = x
        input_net = [net(weight_input[j], x) for j in range(self.J)]
        hidden_layer = [f(input_net[j]) for j in range(self.J)]

        hidden_net = [net(weight_hidden[m], hidden_layer) for m in range(self.M)]
        output = [f(hidden_net[m]) for m in range(self.M)]
        return input_net, hidden_layer, hidden_net, output

    def calculate_error(self, t, output, input_net, hidden_net):
        """
        Оценка ошибок нейронов выходного и скрытого слоя
        :param t: желаемый выход
        :param output: получившийся выход
        :param input_net: комбинированный вход на нейроны скрытого слоя
        :param hidden_net: комбинированный вход на нейронны выходного слоя
        :return: ошибки выходного и скрытого слоя
        """

        output_error = [self.df(hidden_net[m]) * (t[m] - output[m]) for m in range(self.M)]
        hidden_error = [self.df(input_net[j]) *
                        sum(self.weight_hidden[m][j] * output_error[m] for m in range(self.M))
                        for j in range(self.J)]

        return output_error, hidden_error

    def weight_correcting(self, output_error, hidden_error, hidden_layer):
        """
        Настройка весов
        :param output_error: ошибка выходного слоя
        :param hidden_error: ошибка скрытого слоя
        :param hidden_layer:
        :return:
        """

        self.weight_hidden = [[self.weight_hidden[y][x] + self.learn_k * output_error[y] * hidden_layer[x]
                               for x in range(self.J)] for y in range(self.M)]

        self.weight_input = [[self.weight_input[y][x] + self.learn_k * hidden_error[y] * self.x[x]
                              for x in range(self.N)] for y in range(self.J)]

    def net(self, weight, x, w0=1):
        """
        :param weight: весовые коэффициенты
        :param x: входной сигнала
        :param w0: вес смещения
        :return: Комбинированный вход нейрона
        """
        return sum(weight[i] * x[i]
                   for i in range(len(weight))) + w0

    def learning(self):
        """
        Процесс обучения
        """

        error = 1
        self.errors.append(error)
        while error > 0.04:
            input_net, hidden_layer, hidden_net, output = self.calculate_output(self.x)
            # print("CALCULATE OUTPUT", input_net, hidden_layer, hidden_net, output)

            output_error, hidden_error = self.calculate_error(self.t, output, input_net, hidden_net)
            # print("CALCULATE ERROR", output_error, hidden_error)

            error = self.error(output, self.t)
            self.errors.append(error)
            print("Ошибка({}) = {}, выход = {}".format(self.epoch, error, output))

            self.epoch += 1
            self.weight_correcting(output_error, hidden_error, hidden_layer)
            # print("WEIGHT CORRECTING", self.weight_input, self.weight_hidden)

    def error(self, y, y_true):
        """
        Функция суммарной среднеквадратичной ошибки
        :param y: спрогнозированные значения
        :param y_true: реальные значения
        :return: Суммарная среднеквадратичная ошибка
        """
        return math.sqrt(sum((y[i] - y_true[i]) ** 2 for i in range(len(y))))
