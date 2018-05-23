#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Вариант 16

import sys

from matplotlib import pyplot
from network import Network

import math


def f(net):
    """
    Фунция активации
    :param net: комбинированный вход
    """
    return (1 - math.exp(-net)) / (1 + math.exp(-net))


def df(net):
    """
    Производная функции активации
    :param net: комбинированный вход
    """
    f = (1 - math.exp(-net)) / (1 + math.exp(-net))
    return (1 - f ** 2) / 2


def start(N, J, M, x, t):
    """
    Функция запуска обучающего цикла у МНС
    """

    learn_k = 0.9
    network = Network(N, J, M, x, t, learn_k, f, df)

    network.learning()
    plot(network.errors)

def plot(errors):
    """
    :param errors: значения ошибки на эпохах
    """
    pyplot.plot(errors)
    pyplot.grid(True)
    pyplot.savefig('errors.png')
    pyplot.show()


if __name__ == '__main__':

    start_text = '*********************************** Лабораторная 6. **********************************' \
                 '\n Изучение алгоритма обратного распространения ошибки (метод Back Propagation) ' \
                 '\nВведите команду:' \
                 '\n   start                -- начать обучение' \
                 '\n   help                 -- вывод списка комманд' \
                 '\n   exit                 -- выход из программы' \
                 '\n'

    print(start_text)

    # Размер входного слоя
    N = 2

    # Размер скрытого слоя
    J = 1

    # Размер выходного слоя
    M = 2

    # Входной вектор
    x = [1, 2]

    # Целевой вектор
    t = [-1, -1]

    while 1:
        command = input()

        if command == 'start':
            start(N, J, M, x, t)
            print(start_text)
        elif command == 'help':
            print(start_text)
        elif command == '' or command == 'exit':
            break
        else:
            print('Некорректный ввод!')
