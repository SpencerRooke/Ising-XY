# -*- coding: utf-8 -*-
N = 512 #system size = N^2
J = 1 #Interaction strength. Note: Tau =/= Temp
Tau = 1 #Some of the exact constants are wrong

import pyopencl as cl
import numpy as np

rng = np.random.default_rng()
cl.PYOPENCL_COMPILER_OUTPUT=1
context = cl.create_some_context()
queu = cl.CommandQueue(context)

#Load .cl file and create compute program
file = open("Ising.cl", 'r')
kernelString = "".join(file.readlines())
program = cl.Program(context,kernelString).build()

#setup the ising mesh
A = np.random.choice([-1,1], size=int(N**2/2)).astype(np.int32)
B = np.random.choice([-1,1], size=int(N**2/2)).astype(np.int32)
mem_A = cl.Buffer(context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = A)
mem_B = cl.Buffer(context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = B)

def compare(which = 0,Tau = 0 ): #change A,B Globally. Tau is proportional to a unitless Temp
    first_comp = np.float32(np.exp(-J*2/Tau)) #exponentials calculated only once is better
    Rand = rng.random(int(N**2/2),dtype = np.float32)
    mem_Rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = Rand)
    
    compare = program.Comparison
    compare.set_scalar_arg_dtypes([None,None,None,np.uint32,np.uint32,np.float32])
    compare(queu,A.shape,None,mem_A,mem_B,mem_Rand, N, np.int32(which), first_comp)
    
    queu.finish()
    if(which == 0):
        cl.enqueue_copy(queu,A,mem_A)
    else:
        cl.enqueue_copy(queu,B,mem_B)

def combine():
    outArr = np.zeros((N,N))
    for i in range(0,int(N),2):
        k = int(i/2)
        outArr[i][::2] = A[N*k:N*k+N][::2]
        outArr[i+1][1::2] = A[N*k:N*k+N][1::2]
        
        outArr[i][1::2] = B[N*k:N*k+N][1::2]
        outArr[i+1][::2] = B[N*k:N*k+N][::2]
    return outArr
##############################################################################        
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig1=plt.figure(figsize=(32,32))
fig1.tight_layout()
ax=fig1.gca()
ax.set_xticks([])
ax.set_yticks([])
X, Y = np.meshgrid(range(N), range(N))

for i in range(1000):compare((i%2)*1,Tau)
ax.pcolormesh(combine(), cmap=plt.cm.RdPu)
fig1.savefig("example.png")
'''
#Although the simulation is fast enough to be run in real time,
#plotting the animation with matplotlib is very slow (to slow for real time plotting). 
#Revisiting with opengl later could definitely makes this a real time simulation
ims = []
for i in range(3000):
    compare((i%2)*1,Tau)
    ims.append([ax.pcolormesh(combine(), cmap=plt.cm.RdPu)])
    
im_ani = animation.ArtistAnimation(fig1, ims)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=12800)
im_ani.save('im.mp4', writer=writer)
'''