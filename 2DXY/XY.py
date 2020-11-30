# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
cl.PYOPENCL_COMPILER_OUTPUT=1

class XY: #initializes offset lattices A,B for 2D Classical XY problem
    
    def __init__(self,N):
        self.N = N
        self.rng = np.random.default_rng()
        self.context = cl.create_some_context()
        self.queu = cl.CommandQueue(self.context)

        file = open("XY.cl", 'r')
        kernelString = "".join(file.readlines())
        self.program = cl.Program(self.context,kernelString).build()
        #Init with random values between -pi and pi
        self.A = ((np.random.random(size=int(N**2/2))-1/2)*2*np.pi).astype(np.float32)
        self.B = ((np.random.random(size=int(N**2/2))-1/2)*2*np.pi).astype(np.float32)
        self.mem_A = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.A)
        self.mem_B = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.B)

    def reset(self):
        N = self.N
        self.A = ((np.random.random(size=int(N**2/2))-1/2)*2*np.pi).astype(np.float32)
        self.B = ((np.random.random(size=int(N**2/2))-1/2)*2*np.pi).astype(np.float32)
        
    def combine(self):#TODO: write as kernel to speed up animation
        A,B = self.A, self.B
        N = self.N
        outArr = np.zeros((N,N))
        for i in range(0,int(N),2):
            k = int(i/2)
            outArr[i][::2] = A[N*k:N*k+N][::2]
            outArr[i+1][1::2] = A[N*k:N*k+N][1::2]
            
            outArr[i][1::2] = (B[N*k:N*k+N][1::2] + 2*np.pi) % (2 * np.pi) - np.pi
            outArr[i+1][::2] = (B[N*k:N*k+N][::2]  + 2*np.pi) % (2 * np.pi) - np.pi
        return outArr
    
##############################################################################
    def compare(self,which = 0,Tau = 0): #change A,B Globally. Tau is proportional to a unitless Temp
        A,B,mem_A,mem_B = self.A, self.B, self.mem_A, self.mem_B
        context,queu = self.context, self.queu
        N = self.N
        
        #unlike Ising, can't avoid calculating exponential before the parallel compute
        Rand = self.rng.random(int(N**2/2),dtype = np.float32)
        mem_Rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = Rand)
        dPhi = ((np.random.random(size=int(N**2/2))-1/2)*2*np.pi).astype(np.float32)
        mem_dPhi = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = dPhi)
        compare = self.program.Comparison
        
        #TODO: Need to rewrite the kernel!###
        compare.set_scalar_arg_dtypes([None,None,None,None,np.float32,np.uint32,np.uint32])
        compare(queu,A.shape,None,mem_A,mem_B,mem_Rand,mem_dPhi,np.float32(Tau), N, np.int32(which)) 
        
        queu.finish()
        if(which == 0):
            cl.enqueue_copy(queu,A,mem_A)
        else:
            cl.enqueue_copy(queu,B,mem_B)
    
    def animate(self,frames = 300,framesteps = 1,TauRange = [.5,.5],FPS = 30,quiver = False, name = 'outXY.mp4'):
        #TODO: write in OpenGl for realtime viewing, this is really slow
        #self.reset()
        ims = []
        
        fig1=plt.figure(figsize=(64,64))
        fig1.tight_layout()
        ax=fig1.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        if(quiver):ax.set_facecolor('.55')
        TauSet = np.linspace(TauRange[0],TauRange[1],num = frames)
        for i in range(frames):
            for mcSteps in range(framesteps):self.compare((i%2)*1,TauSet[i])
            cur_out = self.combine()
            if(quiver):
                U = np.cos(cur_out)
                V = np.sin(cur_out)
                ims.append([ax.quiver(U,V,cur_out,pivot = 'middle', cmap=plt.cm.hsv)])
            else:
                ims.append([ax.pcolormesh(cur_out, cmap=plt.cm.hsv,vmin = -np.pi,vmax = np.pi)])
        im_ani = animation.ArtistAnimation(fig1, ims)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=12800)
        im_ani.save(name, writer=writer)                