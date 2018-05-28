#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Вариант 16

import sys

from matplotlib import pyplot
import math

from perceptron import Perceptron


def f(t):
    """
    :param t: Аргумент функции
    :return: значение заданной по варианту функции от t
    """
    return math.log(t) - 1


def test_size_window():
    """
    Вычисления
    """
    for i in range(3, 17):
        print(i)
        try:
            start(i)

        except:
            print("Learning is impossible")

        print()


def start(a=1, b=5, N=20, p=4, learn_k=0.3, n=100):
    step = (b - a) / 20

    # Вычисление реальных значений функции
    x = []
    y = []
    i = a
    while i <= (2 * b - a):
        x.append(i)
        y.append(f(i))
        i += step

    X = []
    M = len(x)
    for i in range(M - p):
        X.append(y[i:p + i])

    perceptron = Perceptron(learn_k, X, y, p, n)
    forecast_y, errors, k = perceptron.learning()

    pyplot.plot(x[4:], forecast_y)
    pyplot.plot(x, y)
    pyplot.show()

    pyplot.plot([i for i in range(k)], errors)
    pyplot.show()


def plot(x_real, y_real, x_predicted, y_predicted):
    """
    Строит два графика: реальной функции и спрогнозированной
    :param x_real: лист аргументов функции
    :param y_real: лист значений функции
    :param x_predicted: лист спрогнозированных аргументов
    :param y_predicted: лист спрогнозированных значений
    """
    pyplot.plot(x_real, y_real)
    pyplot.plot(x_predicted, y_predicted)
    pyplot.grid(True)
    pyplot.savefig('result.png')
    pyplot.show()


if __name__ == '__main__':

    start_text = '*********************************** Лабораторная 3. **********************************' \
                 '\n Применение однослойной нейронной сети с линейной функцией активации для прогнозирования временных рядов ' \
                 '\nВведите команду:' \
                 '\n   start [n=3000] [learn_k=0.3] -- начать обучение, n - опционально, количество эпох, learn_k - норма обучения' \
                 '\n   help                         -- вывод списка комманд' \
                 '\n   exit                         -- выход из программы' \
                 '\n'

    print(start_text)

    # Интервал
    a = 1
    b = 5

    # Количество точек
    N = 20

    # Количество нейронов
    p = 4

    # Значение нормы
    learn_k = 0.3

    while 1:
        command = input()

        if command.split()[0] == 'start':
            n = 100
            if len(command.split()) > 2:
                n = command.split()[1]
                learn_k = command.split()[2]
            elif len(command.split()) > 1:
                n = command.split()[1]

            start(a, b, N, p, float(learn_k), int(n))

        elif command == 'plot':
            break
        elif command == 'help':
            print(start_text)
        elif command == '' or command == 'exit':
            break
        else:
            print('Некорректный ввод!')
