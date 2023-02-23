#Verify the relation between area and resistance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
import scipy.optimize as opt
import math


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

#functions that describe Area-Resistance theoretical relation
def ares(x, a, b):
    return a+b/x
def lin (x, a, b):
    return a+b*x

#Read and fit the resistances to verify the relation(R=a+b/A) with area
Area,R2_b,R2_t,errR2_b,errR2_t = read('/Users/cippo/Desktop/tesi/data/w6/w6ares.csv')

p0 = [2.4,28]
p1 = [4.95,26.7]
param_b, param_cov_b = opt.curve_fit(ares, Area, R2_b, p0, errR2_b)
param_t, param_cov_t = opt.curve_fit(ares, Area, R2_t, p1, errR2_t)

print('a+b/x, where:')
print('a=',param_b[0])
print('b=',param_b[1])
print('c+d/x, where:')
print('c=',param_t[0])
print('d=',param_t[1])

#Plot Area-Resistance (4 point probe)
plt.figure(1)
plt.plot(Area,R2_b,marker='.', markersize=0.2, label='Bottom')
plt.plot(Area,R2_t,marker='.', markersize=0.2, label='TOP')
plt.scatter(Area,R2_b,label='Bottom')
plt.scatter(Area,R2_t,label='TOP')
#plt.errorbar(Area, R2_b, yerr=err_R2_b, fmt='o', markersize=1)
#plt.errorbar(Area, R2_t, yerr=err_R2_b, fmt='o', markersize=1)
plt.plot(Area,param_b[0]+param_b[1]/Area,'r', label='Fitted Line Bottom')
plt.plot(Area,param_t[0]+param_t[1]/Area,'g', label='Fitted Line Top')
plt.xlabel('Area($\mu$m$^2$)')
plt.ylabel('Resitance(Ohm)')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()


#Plot (1/Area)-Resistance(4pp) to have linear relation fit with lingress
area = 1/Area
plt.figure(2)
plt.scatter(area,R2_b,s=0.5)
plt.scatter(area,R2_t,s=0.5)
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(area,R2_b)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(area,R2_t)
#print('lingress intercept =',intercept1)
#print('lingress slope =', slope1)
#print('lingress intercept =',intercept2)
#print('lingress slope =', slope2)
plt.plot(area,intercept1 + slope1*area,'r', label='Fitted Line' )
plt.plot(area,intercept2 + slope2*area,'g', label='Fitted Line' )
plt.xlabel('$\\frac{1}{Area(\mu m^2)}$')
plt.ylabel('Resitance(Ohm)')
plt.legend()


#Plot (1/Area)-Resistance (4pp) fit with curvefit to consider error
p2=[intercept1, slope1]
p3=[intercept2, slope2]
plt.figure(3)
plt.errorbar(area, R2_b, yerr=errR2_b, fmt='o', markersize=1)
plt.errorbar(area, R2_t, yerr=errR2_t, fmt='o', markersize=1)
param1, param_cov1 = opt.curve_fit(lin, area, R2_b, p2, errR2_b )
param2, param_cov2 = opt.curve_fit(lin, area, R2_t, p3, errR2_t )
print('curvefit intercept =',param1[0])
print('curvefit slope =',param1[1])
print('curvefit intercept =',param2[0])
print('curvefit slope =',param2[1])
plt.plot(area, param1[0]+param1[1]*area,'r', label='Fitted Line Bottom' )
plt.plot(area, param2[0]+param2[1]*area,'g', label='Fitted Line TOP' )
plt.xlabel('$\\frac{1}{Area(\mu m^2)}$')
plt.ylabel('Resitance(Ohm)')
plt.legend()

plt.show()