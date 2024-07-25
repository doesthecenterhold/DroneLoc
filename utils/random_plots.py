import matplotlib.pyplot as plt
import numpy as np

labels = ['sift', 'superglue', 'lightglue', 'must3r']

x = np.array([1, 2, 3, 4])
y = np.array([6316.237515673522, 991.6058157456849, 2784.2651384938695, 1841.0008174497104]) # means
e = np.array([1744.9897581764835, 205.3123310674831, 517.6778538234265, 175.85332826257007]) # stds

plt.errorbar(x, y, e, linestyle='None', marker='o')
plt.xticks(ticks=[1,2,3,4], labels=labels)

plt.show()