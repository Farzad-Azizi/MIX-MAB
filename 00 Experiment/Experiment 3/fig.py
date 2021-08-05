"""
Created  on Thursday, 5 August 2021
Farzad Azizi 
Email: farzad.azizi@aut.ac.ir 

Benyamin Teymuri
Email: benyamin.teymuri@aut.ac.ir 

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker


# Create the figure and two axes (two rows, one column)
fig, ax1 = plt.subplots(1, 1)
plt.grid()

# Share the x-axis for both the axes (ax1, ax2)
ax2 = ax1.twinx()


a1, a2, a3 = np.loadtxt('ADR-MAB.csv', delimiter=' ', unpack=True)
b1, b2, b3 = np.loadtxt('LoRa-MAB.csv', delimiter=' ', unpack=True)


#PDR
function1 = ax1.plot(a2, a1, '-g', label='ADR-MAB') 
function2 = ax1.plot(b2, b1, '--g', label='LoRa-MAB') 


#ENG
function3 = ax2.plot(a2, a3, '-r', label='ADR-MAB') 
function4 = ax2.plot(b2, b3, '--r', label='LoRa-MAB') 




# Create the legend by first fetching the labels from the functions
functions =  function1 + function2 + function3 + function4
labels = [f.get_label() for f in functions]
plt.legend(functions, labels, loc="lower center", bbox_to_anchor=(0.5,+1), ncol=4)

# Add x-label (only one, since it is shared) and the y-labels
ax1.set_xlabel('Horizon Time (KHours)')
ax1.set_ylabel('Successful Transmission Rate')

#ax1.spines['top'].set_color('g')
ax1.yaxis.label.set_color('g')
ax1.tick_params(axis='y', colors='g')


ax2.set_ylabel('Average Energe Consumption [J]')


#ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')


ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))

ax2.set_ylim([0.17, 0.2])

ax1.set_ylim([0.35, 1])

# Add the title
# plt.title('100N-each end-device can select one of 6 possible SFs from 7 to 12 \n with one sub-channel and the transmission power of 14 dBm',fontsize=10)

# Adjust the figure such that all rendering components fit inside the figure
plt.tight_layout()

# Save the figure
plt.savefig('EX3.png')

plt.show()