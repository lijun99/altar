# Linear Regression synthetic example data generator
import numpy as np
import matplotlib.pyplot as plt

size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a x + b
true_regression_line = true_slope * x + true_intercept
# add noise
y = true_regression_line + np.random.normal(scale=.2, size=size)
# output
np.savetxt('x.txt', x)
np.savetxt('y.txt', y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Generated data and underlying model')
ax.plot(x, y, 'x', label='sampled data')
ax.plot(x, true_regression_line, label='true regression line', lw=2.)
plt.legend(loc=0);
plt.show()

