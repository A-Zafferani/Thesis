#Read data from 4 point probe to obtain (at room temperature) to obtain 
#the resistance, and the critical current as consequence, of the junction 
#the program is ment to read file one by one in order to manually set the
#range of the fit in each file, this process is important for low resistances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
import scipy.optimize as opt
import math


#open the txt file, replace tabs with comma and save the new file as .csv
with open('/Users/cippo/Desktop/tesi/data/w6/D1906/cJJ1w6D1906T-5x20-SQ-1.txt') as f:
    data = f.read().replace('\t', ',')
    print(data, file=open('/Users/cippo/Desktop/tesi/data/w6/D1906/cJJ1w6D1906T-5x20-SQ-1.csv', 'w'))


#read the .csv file and save values in np.array
def read(filename):
    
    dI1 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[0])
    I1 = np.array(dI1).transpose()[0]

    dV1 = pd.read_csv(filename, sep=',',skiprows=(0), usecols=[1])
    V1 = np.array(dV1).transpose()[0]

    dV3 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[2])
    V3 = np.array(dV3).transpose()[0]

    dV4 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[3])
    V4 = np.array(dV4).transpose()[0]

    dI2 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[4])
    I2 = np.array(dI2).transpose()[0] 

    return (I1,V1,V3,V4,I2)

#limit two arrays in a certain range
#automatically select the same indeces for both arrays
def range (arr1,arr2,min,max):

    selected_indices = (arr1 >= min) & (arr1 <= max)
    selected_elements_arr1 = arr1[selected_indices]
    selected_elements_arr2 = arr2[selected_indices]

    return selected_elements_arr1, selected_elements_arr2


#Read I and V than plot and fit to obtain resistance
#V1 variates in time, I1-V1 corresponds to 2 point probe, while I1-(V3-V4) corresponds to 4 point pobe
I1,V1,V3,V4,I2 = read('/Users/cippo/Desktop/tesi/data/w6/D1906/cJJ1w6D1906T-5x20-SQ-1.csv')
rI1,rV1 = range(I1,V1,0,7.62/(1e4))

slope, intercept, r_value, p_value, std_err = stats.linregress(rI1,rV1)
print('R =',round(slope,2),'Ohm')

plt.figure(1)
plt.scatter(I1,V1)
plt.plot(rI1,intercept + slope*rI1, 'r', label='Fitted Line')
plt.legend()
plt.xlabel('$I_1$(A)')
plt.ylabel('$V_1$(V)')

#4 point probe 
Vd=np.subtract(V3,V4)
#Vd = abs(np.subtract(V3,V4))
rI1,rVd = range(I1,Vd,2.5/(1e9),7.62/(1e2))
slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(rI1,rVd)
print('R2 =',round(slope_b,4),'±',round(std_err_b,4),'Ohm')

plt.figure(2)
plt.plot(rI1,rVd)
plt.plot(rI1,intercept_b + slope_b*rI1, 'r', label='Fitted Line')
plt.legend()
plt.xlabel('$I_1$(A)')
plt.ylabel('$V_d$(V)')

plt.show()