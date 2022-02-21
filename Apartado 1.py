# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:57:32 2020

@author: Media Service
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inter

def fun1(Y,t,beta,nu):
    S = Y[0]
    I = Y[1]
    R = Y[2]
    dYdt = np.array([-beta*S*I,beta*S*I - nu*I,nu*I])
    return dYdt

beta = 0.0022 
nu = 0.44
Y0 = np.array([763,1,0])
t1 = np.linspace(0,14,1000)
sol1 = inter.odeint(fun1,Y0,t1,args=(beta,nu))

plt.plot(t1,sol1[:,0],label='Susceptibles',color = 'g')
plt.plot(t1,sol1[:,1],label='Infectados',color='r')
plt.plot(t1,sol1[:,2],label='Retirados',color='b')
plt.grid()
plt.legend()
plt.title('Modelo SIR')
plt.show()


def fun2(Y,t,beta,nu,mu,B):
    S = Y[0]
    I = Y[1]
    R = Y[2]
    dYdt = np.array([-beta*S*I - mu*S + B,beta*S*I - nu*I - mu*I,nu*I - mu*R])
    return dYdt

B = 50
mu = 0.05
t2 = np.linspace(0,40,10000)
sol2 = inter.odeint(fun2,Y0,t2,args=(beta,nu,mu,B))

plt.title('Modelo SIR con nacimientos y muertes')
plt.plot(t2,sol2[:,0],'g',label='susceptibles')
plt.plot(t2,sol2[:,1],'r',label='infectados')
plt.plot(t2,sol2[:,2],'b',label='retirados')
plt.legend()
plt.grid()
plt.show()

def fun3(Y,t,beta,nu,mu,B,fi1,fi2):
    S = Y[0]
    I = Y[1]
    R = Y[2]
    dYdt = np.array([-beta*S*I - mu*S + B - fi1*B - fi2*S,beta*S*I - nu*I - mu*I,nu*I - mu*R + fi1*B + fi2*S])
    return dYdt

fi1 = 0.
fi2 = 0.1
t3 = np.linspace(0,40,10000)
sol3 = inter.odeint(fun3,Y0,t3,args=(beta,nu,mu,B,fi1,fi2))

plt.title('Modelo SIR con inmunidad')
plt.plot(t2,sol3[:,0],'g',label='susceptibles')
plt.plot(t2,sol3[:,1],'r',label='infectados')
plt.plot(t2,sol3[:,2],'b',label='retirados')
plt.legend()
plt.grid()
plt.show()