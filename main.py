import matplotlib.pyplot as plt
import random

N = 50

x = [random.randint(0, 70) for i in range(N)]
y = [random.randint(20, 90) for i in range(N)]
a = [random.randint(50, 99) for i in range(N)]
b = [random.randint(0, 30) for i in range(N)]

plt.scatter(x, y, c=[x[i] * y[i] for i in range(N)])
plt.scatter(a, b, c=[a[i] * b[i] for i in range(N)], cmap="cool")
plt.show()

