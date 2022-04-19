#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

columns = ['Farrah', 'Fred', 'Felicia']
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]
width = 0.5
fig, ax = plt.subplots()
ax.bar(columns, apples, width, label='apples', color='red')
ax.bar(columns, bananas, width, bottom=fruit[0],
       label='bananas', color='yellow')
ax.bar(columns, oranges, width, bottom=fruit[1] + fruit[0],
       label='oranges', color='#ff8000')
ax.bar(columns, peaches, width, bottom=fruit[2] + fruit[1] + fruit[0],
       label='peaches', color='#ffe5b4')
ax.set_ylabel('Quantity of Fruit')
ax.set_ylim([0, 80])
ax.set_title('Number of Fruit per Person')
ax.legend()
plt.show()
