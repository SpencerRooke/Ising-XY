# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
cl.PYOPENCL_COMPILER_OUTPUT=1

class Ising: #initializes offset lattices A,B for d dimensional cubic ising problem (size N^d)
    
    def __init__(self,N,J,dimension=2, oclString = "Ising.cl"):
        self.N, self.J = N,J 
        self.D = dimension
        self.rng = np.random.default_rng()
        self.context = cl.create_some_context()
        self.queu = cl.CommandQueue(self.context)

        #Load .cl file and create compute program
        file = open(oclString, 'r')
        kernelString = "".join(file.readlines())
        self.program = cl.Program(self.context,kernelString).build()
        
        self.A = np.random.choice([-1,1], size=int(N**dimension/2)).astype(np.int32)
        self.B = np.random.choice([-1,1], size=int(N**dimension/2)).astype(np.int32)
        self.mem_A = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.A)
        self.mem_B = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.B)

    def reset(self):
        N, D = self.N, self.D
        self.A = np.random.choice([-1,1], size=int(N**D/2)).astype(np.int32)
        self.B = np.random.choice([-1,1], size=int(N**D/2)).astype(np.int32)
        
    def mag(self): #return mag, which can be used to estimate T_c
        total_mag = np.sum(self.A)+np.sum(self.B)
        return total_mag    
    
    def Energy(self):#Not Tested, maybe doesn't work. Need to check dtypes in kernel call
        A,B,mem_A,mem_B = self.A, self.B, self.mem_A, self.mem_B
        context,queu = self.context, self.queu
        N,J = self.N,self.J
        
        currentE = np.int32(0)
        mem_E = cl.Buffer(context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = currentE)
        
        inter_E = self.program.Energy
        inter_E.set_scalar_arg_dtypes([None,None,None,np.uint32])
        inter_E(queu,np.int32,None,mem_A,mem_B, mem_E, N)
        queu.finish()
        cl.enqueue_copy(queu,currentE,mem_E)
        return currentE*J #maybe missing a factor
##############################################################################        
class Ising2D(Ising):

    def compare(self,which = 0,Tau = 0): #change A,B Globally. Tau is proportional to a unitless Temp
        A,B,mem_A,mem_B = self.A, self.B, self.mem_A, self.mem_B
        context,queu = self.context, self.queu
        N,J = self.N,self.J
        
        first_comp = np.float32(np.exp(-J*2/Tau)) #exponentials calculated only once is better
        Rand = self.rng.random(int(N**2/2),dtype = np.float32)
        mem_Rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = Rand)
        
        compare = self.program.Comparison
        compare.set_scalar_arg_dtypes([None,None,None,np.uint32,np.uint32,np.float32])
        compare(queu,A.shape,None,mem_A,mem_B,mem_Rand, N, np.int32(which), first_comp)
        
        queu.finish()
        if(which == 0):
            cl.enqueue_copy(queu,A,mem_A)
        else:
            cl.enqueue_copy(queu,B,mem_B)

    def combine(self):#TODO: write as kernel to speed up animation
        A,B = self.A, self.B
        N = self.N
        outArr = np.zeros((N,N))
        for i in range(0,int(N),2):
            k = int(i/2)
            outArr[i][::2] = A[N*k:N*k+N][::2]
            outArr[i+1][1::2] = A[N*k:N*k+N][1::2]
            
            outArr[i][1::2] = B[N*k:N*k+N][1::2]
            outArr[i+1][::2] = B[N*k:N*k+N][::2]
        return outArr
    
    def animate(self,frames = 3000,TauRange = [1,1],FPS = 60):
        #TODO: write in OpenGl for realtime viewing
        self.reset()
        ims = []
        
        fig1=plt.figure(figsize=(32,32))
        fig1.tight_layout()
        ax=fig1.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        
        TauSet = np.linspace(TauRange[0],TauRange[1],num = frames)
        for i in range(frames):
            self.compare((i%2)*1,TauSet[i])
            ims.append([ax.pcolormesh(self.combine(), cmap=plt.cm.RdPu)])
            
        im_ani = animation.ArtistAnimation(fig1, ims)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=12800)
        im_ani.save('out.mp4', writer=writer)

##############################################################################
class Ising3D(Ising):#make sure to invoke with dimension=3 and oclString = "Ising_3.cl"
    
    def compare(self,which=0,Tau=0):
        A,B,mem_A,mem_B = self.A, self.B, self.mem_A, self.mem_B
        context,queu = self.context, self.queu
        N,J = self.N,self.J
        
        first_comp = np.float32(np.exp(-J*2/Tau))
        Rand = self.rng.random(int(N**3/2),dtype = np.float32)
        mem_Rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = Rand)
        
        update = self.program.Update
        update.set_scalar_arg_dtypes([None,None,None,np.uint32,np.uint32,np.float32]) #<---
        update(queu,A.shape,None,mem_A,mem_B,mem_Rand, N, np.int32(which), first_comp) #<---
        
        queu.finish()
        if(which == 0):
            cl.enqueue_copy(queu,A,mem_A)
        else:
            cl.enqueue_copy(queu,B,mem_B)
    
    def combine():pass #not needed if not plotting
    #The visualizations with matplotlib for 3D weren't great, so I got rid of animate here
    #TODO: Revisit with OpenGL for visualization
        
##############################################################################
Tau = 1 
N = 512
ising = Ising2D(10,J=2)

pArray = []
tArray = []

#for k in range(1024):ising.compare((k%2)*1,1.5)
for i in range(64):
    temp = 3-i*1.5/64
    tArray=tArray+[temp]
    #equilibriate. Should already be close from previous state
    for j in range(512):ising.compare((j%2)*1,temp)
    pArray = pArray + [ising.mag()]

fig1=plt.figure(figsize=(32,32))
fig1.tight_layout()
ax=fig1.gca()
ax.plot(tArray,np.abs(pArray))

