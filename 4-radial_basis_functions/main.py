#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Вариант 16

import sys

from matplotlib import pyplot
import math

from rbf import RBF


def step(net):
    """
    Пороговая функция активации
    """
    return 1 if net >= 0 else 0


def start(f, x, centers, learn_k):
    network = RBF(f, x, centers, step, learn_k)
    epoch, errors = network.learning()

    return epoch, errors


def plot(x, y):
    """
    Строит график
    :param x: лист аргументов функции
    :param y: лист значений функции
    """
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.show()


if __name__ == '__main__':

    start_text = '*********************************** Лабораторная 4. **********************************' \
                 '\n Исследование нейронных сетей с радиальными базисными функциями (RBF) на примере моделирования булевых выражений ' \
                 '\nВведите команду:' \
                 '\n   start                -- начать обучение' \
                 '\n   plot                 -- построить график результатов эксперимента' \
                 '\n   help                 -- вывод списка комманд' \
                 '\n   exit                 -- выход из программы' \
                 '\n'

    print(start_text)

    f = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    x = ((0, 0, 0, 0),
         (0, 0, 0, 1),
         (0, 0, 1, 0),
         (0, 0, 1, 1),
         (0, 1, 0, 0),
         (0, 1, 0, 1),
         (0, 1, 1, 0),
         (0, 1, 1, 1),
         (1, 0, 0, 0),
         (1, 0, 0, 1),
         (1, 0, 1, 0),
         (1, 0, 1, 1),
         (1, 1, 0, 0),
         (1, 1, 0, 1),
         (1, 1, 1, 0),
         (1, 1, 1, 1))

    # Для моей функции из ЛР1 J = 3
    # Центры RBF
    centers = [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]

    # Минимальный набор векторов из ЛР1
    min_x = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
    min_y = [1, 1, 0, 0]

    # Значение нормы
    learn_k = 0.3

    errors = []
    epoch = 0

    while 1:
        command = input()

        if command == 'start':
            epoch, errors = start(f, x, centers, learn_k)

        elif command == 'plot':
            if len(errors) == 0:
                epoch, errors = start(f, x, centers, learn_k)

            plot([i for i in range(epoch)], errors)
        elif command == 'help':
            print(start_text)
        elif command == '' or command == 'exit':
            break
        else:
            print('Некорректный ввод!')
