#take area dose and resistance from different wafers, than subtract the spurious contribution
#than show the dose resistance relation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
import scipy.optimize as opt
import math

#read the .csv file and save values in np.array
def read(filename):
    
    dA = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[0])
    A = np.array(dA).transpose()[0]

    dR2 = pd.read_csv(filename, sep=',',skiprows=(0), usecols=[1])
    R2 = np.array(dR2).transpose()[0]

    dR1 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[2])
    R1 = np.array(dR1).transpose()[0]

    dR4 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[3])
    R4 = np.array(dR4).transpose()[0]

    dR3 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[4])
    R3 = np.array(dR3).transpose()[0]

    dR6 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[5])
    R6 = np.array(dR6).transpose()[0]

    dR5 = pd.read_csv(filename, sep=',', skiprows=(0), usecols=[6])
    R5 = np.array(dR5).transpose()[0] 

    return (A,R2,R1,R4,R3,R6,R5)

#area resistance relation function
def ares(x, a, b):
    return a+b/x
#dose resistance relation function
def dres(x, a, b):
    return a+b*(x**0.5)
def lin (x, a, b):
    return a+b*x
def lin2(x,a):
    return a*x


#read and fit values of area and resistances from the wafer
#w5 has no oxidation, dose w2<w1<w4
A,R2,R1,R4,R3,R6,R5 = read("Arestot copy.csv")

p2=[1,1]
p1=[1,1]
p4=[1,1]
p3=[1,1]
p6=[1,1]
p5=[1,1]

param_2, param_cov_2 = opt.curve_fit(ares, A, R2, p2)
param_1, param_cov_1 = opt.curve_fit(ares, A, R1, p1)
param_4, param_cov_4 = opt.curve_fit(ares, A, R4, p4)
param_3, param_cov_3 = opt.curve_fit(ares, A, R3, p3)
param_6, param_cov_6 = opt.curve_fit(ares, A, R6, p6)
param_5, param_cov_5 = opt.curve_fit(ares, A, R5, p5)

#plot area-resistance fitted relations for the wafers
plt.figure(1)
#plt.scatter(A,R2)
plt.plot(A,param_2[0]+param_2[1]/A, label='Wafer 2')
#plt.scatter(A,R1)
plt.plot(A,param_1[0]+param_1[1]/A, label='Wafer 1')
#plt.scatter(A,R4)
plt.plot(A,param_4[0]+param_4[1]/A, label='Wafer 4')
#plt.scatter(A,R3)
plt.plot(A,param_3[0]+param_3[1]/A, label='Wafer 3')
#plt.scatter(A,R6)
plt.plot(A,param_6[0]+param_6[1]/A, label='Wafer 6')
#plt.scatter(A,R5)
plt.plot(A,param_5[0]+param_5[1]/A, label='Wafer 5')

plt.legend()
plt.xlabel('Area($\mu m^2$)')
plt.ylabel('Resistance(Ohm)')
plt.title('Area Resistance')


#subtract to the oxidised wafer the spurius resistance contribution
#obtained by fitting the wafer w/o oxide
diff2=param_2[0]-param_5[0]+(param_2[1]-param_5[1])/A
diff1=param_1[0]-param_5[0]+(param_1[1]-param_5[1])/A
diff4=param_4[0]-param_5[0]+(param_4[1]-param_5[1])/A
diff3=param_3[0]-param_5[0]+(param_3[1]-param_5[1])/A
diff6=param_6[0]-param_5[0]+(param_6[1]-param_5[1])/A


plt.figure(2)
plt.plot(A,diff2, label='Wafer 2')
plt.plot(A,diff1, label='Wafer 1')
plt.plot(A,diff4, label='Wafer 4')
plt.plot(A,diff3, label='Wafer 3')
plt.plot(A,diff6, label='Wafer 6')


plt.legend()
plt.xlabel('Area($\mu m^2$)')
plt.ylabel('Resistance(Ohm)')
plt.title('Effective Area Resistance')

print('Resistance2=', param_2[0]-param_5[0]+(param_2[1]-param_5[1]) , 'Ohm*um^2')
print('Resistance1=', param_1[0]-param_5[0]+(param_1[1]-param_5[1]), 'Ohm*um^2')
print('Resistance4=', param_4[0]-param_5[0]+(param_4[1]-param_5[1]), 'Ohm*um^2')
print('Resistance4=', param_3[0]-param_5[0]+(param_3[1]-param_5[1]), 'Ohm*um^2')
print('Resistance4=', param_6[0]-param_5[0]+(param_6[1]-param_5[1]), 'Ohm*um^2')

#dose for the wafers 
dose=np.array([3,7.2,28.8,115.2,230.4])
rdose = dose**0.5

diffv2=R2-R5
diffv1=R1-R5
diffv4=R4-R5
diffv3=R3-R5
diffv6=R6-R5

print('Area*resistenza effettiva w2',A*diffv2)
print('Area*resistenza effettiva w1',A*diffv1)
print('Area*resistenza effettiva w4',A*diffv4)
print('Area*resistenza effettiva w3',A*diffv3)
print('Area*resistenza effettiva w6',A*diffv6)

#create and fill a matrix with the resistances, w/o spurious contribution
#each rows contain resistances from same wafer
#each column contain resistances with same area
mat=np.empty([5,10])

for i in range(10):
    mat[0][i] = diffv2[i]
    mat[1][i] = diffv1[i]
    mat[2][i] = diffv4[i]
    mat[3][i] = diffv3[i]
    mat[4][i] = diffv6[i]


#dose-resistance relation fit, in this case are considered
#resistances from different wafer, so different dose, but with same area
d1 = [1,1]
d2 = [1,1]
d3 = [1,1]

paramd_1, paramd_cov_1 = opt.curve_fit(dres, dose, mat[:,2], d1)
print(paramd_1)


plt.figure(3)
plt.scatter(dose,mat[:,2])
plt.plot(dose,paramd_1[0]+paramd_1[1]*(dose**0.5), label='25$\mu m^2$')
plt.xlabel('Dose(mbar*s)')
plt.ylabel('Resistance(Ohm)')
plt.legend()
#plt.title('15$\mu m^2$')


#(dose**0.5)-resistance relation
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(rdose,mat[:,2])

print(intercept1, slope1)

plt.figure(4)
plt.scatter(rdose,mat[:,2])
plt.plot(rdose,intercept1+slope1*rdose, label='15$\mu m^2$')
plt.xlabel('$\sqrt{Dose(mbar*s)}$')
plt.ylabel('Resistance(Ohm)')
plt.legend()


#(dose**0.5)-resistance relation but this time error is considered
eR2,eR1,eR4,eR3,eR6,eR5,n = read('errortot.csv')

error2 = (eR2**2 + eR5**2)**0.5
error1 = (eR1**2 + eR5**2)**0.5
error4 = (eR4**2 + eR5**2)**0.5
error3 = (eR3**2 + eR5**2)**0.5
error6 = (eR6**2 + eR5**2)**0.5

#creating a matrix containg the errors, obtained by fit and error propagation, of the 'mat' matrix
err=np.empty([5,10])

for i in range(10):
    err[0][i] = error2[i]
    err[1][i] = error1[i]
    err[2][i] = error4[i]
    err[3][i] = error3[i]
    err[4][i] = error6[i]


de1=[intercept1,slope1]
de2=[1]


parame_1, parame_cov_1 = opt.curve_fit(lin, rdose, mat[:,0], de1,err[:,0])
parame_2, parame_cov_2 = opt.curve_fit(lin2, rdose, mat[:,0], de2,err[:,0])


print('error=',err[:,0])
print('parame=',parame_1)
print('parame cov=',parame_cov_1)
print('parame2=',parame_2)

plt.figure(5)
plt.errorbar(rdose, mat[:,0], yerr=err[:,0], fmt='o', markersize=5)
plt.plot(rdose,parame_1[0]+parame_1[1]*rdose, label='w intercept')
plt.plot(rdose,parame_2[0]*rdose, label='w/o intercept')
plt.xlabel('$\sqrt{Dose(mbar*s)}$')
plt.ylabel('Resistance(Ohm)')
plt.legend()

#in this case is shown also the 0 oxidatio/resistance to better visualise the obtined relation
mat2=np.empty([6,10])
for i in range(10):
    mat2[0][i] = 0
    mat2[1][i] = diffv2[i]
    mat2[2][i] = diffv1[i]
    mat2[3][i] = diffv4[i]
    mat2[4][i] = diffv3[i]
    mat2[5][i] = diffv6[i]


err2=np.empty([6,10])
for i in range(10):
    err2[0][i] = eR5[i]
    err2[1][i] = error2[i]
    err2[2][i] = error1[i]
    err2[3][i] = error4[i]
    err2[4][i] = error3[i]
    err2[5][i] = error6[i]


dose2=np.array([0,3,7.2,28.8,115.2,230.4])
rdose2=dose2**0.5

de3=[intercept1,slope1]
de4=[1]

parame_3, parame_cov_3 = opt.curve_fit(lin, rdose2, mat2[:,1], de3,err2[:,1])
parame_4, parame_cov_4 = opt.curve_fit(lin2, rdose2, mat2[:,1], de4,err2[:,1])
print('parame3=',parame_3)
print('parame3 cov=',parame_cov_3)
print('parame4=',parame_4)
plt.figure(6)
plt.errorbar(rdose2, mat2[:,1], yerr=err2[:,1], fmt='o', markersize=5)
plt.plot(rdose2,parame_3[0]+parame_3[1]*rdose2, label='w intercept')
plt.plot(rdose2,parame_4[0]*rdose2, label='w/o intercept')
plt.xlabel('$\sqrt{Dose(mbar*s)}$')
plt.ylabel('Resistance(Ohm)')
plt.title('20 $\mu m^2$')
plt.legend()


plt.show()