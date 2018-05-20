#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from kohonen_network import KohonenNetwork

class ClusteringApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Кластеризация с НС Кохонова")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        frame = MainPage(container, self)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


class MainPage(tk.Frame):
    def read_json(self):
        """
        Функция считывания датасета из .json файлов
        """
        json_data = open("libraries.json").read()

        data = json.loads(json_data)
        data = [data['features'][i]['geometry']['coordinates'][0] for i in range(len(data['features']))]
        libraries = pd.DataFrame(data, columns=['x', 'y'])
        libraries['color'] = np.random.rand()
        libraries['size'] = 8

        self.libraries = libraries

        json_data = open("districts.json").read()

        districts_data = json.loads(json_data)
        districts_data = [districts_data['features'][i]['geometry']['coordinates']
                          for i in range(len(districts_data['features']))]

        districts = pd.DataFrame(districts_data, columns=['x', 'y'])
        districts['color'] = 0.6
        districts['size'] = 25

        self.districts = districts

    def clusterHandler(self):
        """
        Функция обработки нажатия на кнопку Кластеризовать
        Перерисовывает граф в соответствии с новыми кластерами
        """
        network = KohonenNetwork(self.libraries[['x', 'y']].values, self.districts[['x', 'y']].values)
        weights = network.calculate_weights()

        colors = np.random.rand(self.districts.shape[0])

        l_data = self.libraries.values
        d_data = self.districts.values

        for i in range(len(d_data)):
            d_data[i][2] = colors[i]

        for i in range(len(l_data)):
            center = weights[i].argmax()
            l_data[i][2] = colors[center]

        self.libraries['color'] = l_data.T[2]
        self.districts['color'] = d_data.T[2]

        self.plot.clear()
        self.plot.scatter(self.libraries.append(self.districts)[['x']], self.libraries.append(self.districts)[['y']],
                          s=self.libraries.append(self.districts)[['size']],
                          c=self.libraries.append(self.districts)[['color']])
        self.plot.axis('off')

        self.canvas.show()

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        button = ttk.Button(self, text="Кластеризовать", command=lambda: self.clusterHandler())
        button.pack()

        self.read_json()

        f = Figure(figsize=(10, 7), dpi=100)
        self.plot = f.add_subplot(111)
        self.plot.scatter(self.libraries.append(self.districts)[['x']], self.libraries.append(self.districts)[['y']],
                          s=self.libraries.append(self.districts)[['size']],
                          c=self.libraries.append(self.districts)[['color']])
        self.plot.axis('off')

        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


np.random.seed(424242)
app = ClusteringApp()
app.mainloop()
