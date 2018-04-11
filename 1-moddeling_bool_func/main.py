#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Вариант 16

import sys

from itertools import combinations
from matplotlib import pyplot

""" 
Возвращает: значение пороговой функции от входа net
"""
def step(net):
    return 1 if net >= 0 else 0

"""
Возвращает: значение сигмоиды от входа net
"""
def sigmoid(net):
    return 1 if 0.5 * (net / (1 + abs(net)) + 1) >= 0.5 else 0

"""
Возвращает: значение производной сигмоиды от входа net
"""
def der_sigmoid(net):
    f = 0.5 * (net / (1 + abs(net)) + 1)
    return 0.5 * ((1 - abs(f)) ** 2)

"""
Возвращает: комбинированный вход
"""
def net(w, x):
    return sum(w[i] * x[i] for i in range(5))

"""
Функция обучения НС

Возвращает: число эпох, массив значений ошибок
"""
def learning(f, x, activ_func, d_activ_func):
    # Начальные веса
    w = [0, 0, 0, 0, 0]

    # Значение нормы
    learn_k = 0.3

    # Реальный выход НС
    y = [activ_func(net(w, x[i])) for i in range(16)]

    # Значение ошибки: расстояние Хэмминга между векторами целевого и реального выхода
    e = sum((f[i] ^ y[i] for i in range(16)))

    e_array = [e]
    print('0 y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], e=%d' % (str(y), w[0], w[1], w[2], w[3], w[4], e))

    # Номер эпохи
    k = 1

    while e > 0:
        # Разница между целевым и реальным выходом
        delta = tuple((f[i] - y[i] for i in range(16)))

        # Коррекция веса
        for i in range(5):
            if d_activ_func != 1:
                w[i] += sum(learn_k * delta[j] * x[j][i] * d_activ_func(net(w, x[i])) for j in range(16))
            else:
                w[i] += sum(learn_k * delta[j] * x[j][i] for j in range(16))

        # Пересчитывание реального выхода НС
        y = [activ_func(net(w, x[i])) for i in range(16)]

        # Пересчитывание ошибки
        e = sum((f[i] ^ y[i] for i in range(16)))
        e_array.append(e)

        print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], e=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], e))

        k += 1
        if k > 100: return -1

    return k - 1, e_array


def index(subset):
    return subset[4] + 2 * subset[3] + 4 * subset[2] + 8 * subset[1]

"""
Функция обучения для подмножеств

Возвращает: число эпох
"""
def finding_min(f, x, activ_func, d_activ_func, subset, flag):
    w = [0, 0, 0, 0, 0]
    learn_k = 0.3

    y = [activ_func(net(w, x[i])) for i in range(16)]

    e = sum((f[i] ^ y[i] for i in range(16)))

    if flag:
        print('0 y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], e=%d' % (str(y), w[0], w[1], w[2], w[3], w[4], e))

    k = 1

    while e > 0:
        delta = tuple((f[i] - y[i] for i in range(16)))

        for i in range(5):
            if d_activ_func != 1:
                w[i] += sum(learn_k * delta[index(subset[j])] * subset[j][i] * d_activ_func(net(w, subset[j])) for j in
                            range(len(subset)))
            else:
                w[i] += sum(learn_k * delta[index(subset[j])] * subset[j][i] for j in range(len(subset)))

        y = [activ_func(net(w, x[i])) for i in range(16)]
        e = sum((f[i] ^ y[i] for i in range(16)))

        if flag:
            print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], e=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], e))

        k += 1
        if k > 100: return -1

    return k - 1


"""
Функция обработки комманды обучения пороговой функцией
"""
def step_command():
    k, _ = learning(f, x, step, 1)
    print('\nСработало за %d эпох' % k)

"""
Функция обработки комманды обучения сигмоидой
"""
def sigmoid_command():
    k, _ = learning(f, x, sigmoid, der_sigmoid)
    print('\nСработало за %d эпох' % k)

"""
Функция обработки комманды нахождения минимального набора
"""
def min_command(mode):
    if mode == 'step':
        activation_function = step
        derivative_activ_func = 1
    elif mode == 'sigmoid':
        activation_function = sigmoid
        derivative_activ_func = der_sigmoid
    else:
        return -1

    # Проходимся от 2 до 16 векторов
    for i in range(2, 16):

        # Всевозможные комбинации i векторов
        all_combinations = list(combinations(x, i))

        print('Перебор вариантов из %d векторов...' % i)

        for subset in all_combinations:

            # Отключаем вывод обучения
            flag = 0
            count = finding_min(f, x, activation_function, derivative_activ_func, subset, flag)

            # Если смогли обучить (кол-во эпох ненулевое), выводим
            if count > 0:
                print('Наборы: %s' % str(subset))
                flag = 1
                finding_min(f, x, activation_function, derivative_activ_func, subset, flag)
                break
        if flag == 1: break


"""
Функция обработки комманды вывода графика
"""
def plot_command(mode):
    if mode == 'step':
        activation_function = step
        derivative_activ_func = 1
    elif mode == 'sigmoid':
        activation_function = sigmoid
        derivative_activ_func = der_sigmoid
    else:
        return -1

    _, errors = learning(f, x, activation_function, derivative_activ_func)
    pyplot.plot(errors)
    pyplot.show()


if __name__ == '__main__':

    f = (1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)

    x = ((1, 0, 0, 0, 0),
         (1, 0, 0, 0, 1),
         (1, 0, 0, 1, 0),
         (1, 0, 0, 1, 1),
         (1, 0, 1, 0, 0),
         (1, 0, 1, 0, 1),
         (1, 0, 1, 1, 0),
         (1, 0, 1, 1, 1),
         (1, 1, 0, 0, 0),
         (1, 1, 0, 0, 1),
         (1, 1, 0, 1, 0),
         (1, 1, 0, 1, 1),
         (1, 1, 1, 0, 0),
         (1, 1, 1, 0, 1),
         (1, 1, 1, 1, 0),
         (1, 1, 1, 1, 1))

    start_text = '*********************************** Лабораторная 1. **********************************' \
                 '\n Исследование однослойных нейронных сетей на примере моделирования булевых выражений ' \
                 '\nВведите команду:' \
                 '\n   step            -- обучение с помощью пороговой функции' \
                 '\n   sigmoid         -- обучение с помощью сигмоиды' \
                 '\n   min-step        -- нахождение минимального набора для пороговой функции' \
                 '\n   min-sigmoid     -- нахождение минимального набора для сигмоиды' \
                 '\n   plot-step       -- построить график для пороговой функции' \
                 '\n   plot-sigmoid    -- построить график для сигмоиды' \
                 '\n   help            -- вывод списка комманд' \
                 '\n   exit            -- выход из программы' \
                 '\n'

    print(start_text)

    while 1:
        command = raw_input()

        if command == 'step':
            step_command()
        elif command == 'sigmoid':
            sigmoid_command()
        elif command == 'min-step':
            min_command('step')
        elif command == 'min-sigmoid':
            min_command('sigmoid')
        elif command == 'plot-step':
            plot_command('step')
        elif command == 'plot-sigmoid':
            plot_command('sigmoid')
        elif command == 'help':
            print(start_text)
        elif command == '' or command == 'exit':
            break
        else:
            print('Некорректный ввод!')