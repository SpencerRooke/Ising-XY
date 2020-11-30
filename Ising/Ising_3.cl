#define id(i,j,k,N) ( (((i)%(N/2))*N)*N + ((j)%N)*N + ((k)%N) )

//TODO: Doesn't work, probably an index problem
__kernel void Update
(__global int* A, __global int* B, __global float* rand, const unsigned int N,int which, float comp_one)
{
    float comp_two = comp_one*comp_one; //handle exponentiation outside of repeated comp (python side)
    float comp_three = comp_one*comp_one*comp_one;
    int comp_ijk;
    float tempVal;
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1); //will need smaller system sizes than 2D with this config

	if(which == 0){ //update A mat (reads from B mat)
    	if(i<N/2 && j<N){
    		for (int k = 0; k<N; k++){
    		//will move to seperate function when I confirm this works
    		comp_ijk = B[id(i,j,k,N)] +
    		B[id(i,j+1,k,N)] + B[id(i,j-1,k,N)] +
    		B[id(i,j,k+1,N)] + B[id(i,j,k-1,N)] +
    		((j+k)%2==1) * B[id(i-1,j,k,N)] + 
    		((j+k)%2==0) * B[id(i+1,j,k,N)];   		
    		
            comp_ijk = A[id(i,j,k,N)]*comp_ijk;
            
            tempVal = 0 + (comp_ijk == 2)*comp_one + (comp_ijk == 4)*comp_two + (comp_ijk == 6)*comp_three;
            //Below may change
            A[id(i,j,k,N)] = (-2*(tempVal == 0 || tempVal > rand[id(i,j,k,N)]) + 1) * A[id(i,j,k,N)]; 
    		}
    	}
    } else { //update B mat (reads from A mat)
        if(i<N/2 && j<N){
    		for (int k = 0; k<N; k++){
    		//will move to seperate function when I confirm this works
    		comp_ijk = A[id(i,j,k,N)] +
    		A[id(i,j+1,k,N)] + A[id(i,j-1,k,N)] +
    		A[id(i,j,k+1,N)] + A[id(i,j,k-1,N)] +
    		((j+k)%2==1) * A[id(i+1,j,k,N)] + 
    		((j+k)%2==0) * A[id(i-1,j,k,N)];   		
    		
            comp_ijk = B[id(i,j,k,N)]*comp_ijk;
            
            tempVal = 0 + (comp_ijk == 2)*comp_one + (comp_ijk == 4)*comp_two + (comp_ijk == 6)*comp_three;
            //Below may change
            B[id(i,j,k,N)] = (-2*(tempVal == 0 || tempVal > rand[id(i,j,k,N)]) + 1) * B[id(i,j,k,N)]; 
    		}
        }
    }
}

__kernel void Magnetization //np.sum is probably just as fast
(__global int* A, __global int* B, const unsigned int N,  __global int* Mag)
{
    int comp = 0;
    unsigned int i = get_global_id(0);
    if(i<N*N/2){
        comp = comp+A[i]+B[i];
    Mag = comp;
    }
}

__kernel void Energy 
(__global int* A, __global int* B, __global int* Energy, const unsigned int N)
{
    int comp = 0;
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    
    if(i<N/2 && j<N){
    	for (int k = 0; j<N; j++){
    	comp = comp + -1*A[id(i,j,k,N)] * 
    	(B[id(i,j+1,k,N)]               + 
	    ((j+k)%2==0) * B[id(i,j,k,N)]   + 
    	((j+k)%2==1) * B[id(i+1,j,k,N)] +
    	B[id(i,j,k+1,N)]);
    	
    	comp = comp + -1*B[id(i,j,k,N)] * 
    	(A[id(i,j+1,k,N)]               + 
	    ((j+k)%2==0) * A[id(i+1,j,k,N)] + 
        ((j+k)%2==1) * A[id(i,j,k,N)]	+
        A[id(i,j,k+1,N)]);
    	}
    
    comp = comp/4;
    Energy = comp;
    //factor here may be wrong
    }
}



