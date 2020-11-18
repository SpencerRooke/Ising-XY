# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Ising import Ising2D

matplotlib.rcParams.update({'font.size': 64})
matplotlib.rcParams['axes.linewidth'] = 2

fig1,fig2=plt.figure(figsize=(32,32)),plt.figure(figsize=(32,32))
fig1.tight_layout()
ax1, ax2 = fig1.gca(), fig2.gca()

ax1.tick_params(which='both',direction="in",length=7,width=3)
ax1.tick_params(which='major',length=8)
ax1.tick_params(axis="x", pad=10)
ax2.tick_params(which='both',direction="in",length=7,width=3)
ax2.tick_params(which='major',length=8)
ax2.tick_params(axis="x", pad=10)

ax1.set_ylabel("m") 
ax1.set_xlabel(r"$\tau$")
ax2.set_ylabel(r"$\chi$") 
ax2.set_xlabel(r"$\tau$")

T_c = []
NArray = [16,64,256]
shapeArray = ['^-','s-','d-']
for r in range(len(NArray)):
    Num = NArray[r]
    ising = Ising2D(Num,J=1)
    tArray = np.linspace(1.5,3,150)
    pArray,chi = [],[]
    for j in range(1024):ising.compare((j%2)*1,1.5)
    for temp in tArray:
        mag,mag2=0,0
        #equilibriate. Should already be close from previous state
        for j in range(512):ising.compare((j%2)*1,temp)
        for k in range(1024): #Average over a few steps
            ising.compare((k%2)*1,temp)
            currentMag = np.abs(ising.mag())/1024
            mag += currentMag
            mag2 += 1024*currentMag**2
        pArray = pArray + [mag/(Num**2)]
        chi = chi + [(mag2 - mag**2)/(Num**2 * temp)]
    ax1.plot(tArray,pArray,shapeArray[r],ms = 30,linewidth = 5, label = "N = "+str(Num))
    ax2.plot(tArray,chi,shapeArray[r],ms = 30,linewidth = 5, label = "N = "+str(Num))
    T_c = T_c + [tArray[np.argmax(chi)]]
    del ising
ax1.legend()
ax2.legend()
fig1.savefig('Mag.png')
fig2.savefig('Susc.png')
