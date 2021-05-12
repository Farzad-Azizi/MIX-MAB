import numpy as np
import matplotlib.pyplot as plt

# Create the figure and two axes (two rows, one column)
fig, ax1 = plt.subplots(1, 1)

# Share the x-axis for both the axes (ax1, ax2)
ax2 = ax1.twinx()

x1, x2, x3, x4 = np.loadtxt('20.csv', delimiter=' ', unpack=True)
x5, x6, x7, x8 = np.loadtxt('100.csv', delimiter=' ', unpack=True)

function1 = ax1.plot(x2, x3, '--b', marker='.', label='20-PDR')
function2 = ax2.plot(x2, x4, 'r', marker='*', label='20-Energy')

function3 = ax1.plot(x6, x7, '--g', marker='.', label='100-PDR')
function4 = ax1.plot(x6, x8, 'y', marker='*', label='100-Energy')

# Create the legend by first fetching the labels from the functions
functions = function1 + function2+ function3+ function4
labels = [f.get_label() for f in functions]
plt.legend(functions, labels, loc=0)

# Add x-label (only one, since it is shared) and the y-labels
ax1.set_xlabel('$simtime$')
ax1.set_ylabel('$Successful-Transmission-Rate$')
ax2.set_ylabel('$Average-Energe$')

# Add the title
plt.title('PDR and Energy consumption')

# Adjust the figure such that all rendering components fit inside the figure
plt.tight_layout()

# Save the figure
plt.savefig('result.png')
