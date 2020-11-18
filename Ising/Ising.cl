#define id(x,y,N) ( ((x)%(N/2))*N + ((y)%N) )

//this can definitely be implemented better, will revisit when better with open cl
__kernel void Comparison
(__global int* A, __global int* B, __global float* rand, const unsigned int N,int which, float comp_one)
{
    float comp_two = comp_one*comp_one;
    int comp_ij;
    float tempVal;
	unsigned int i = get_global_id(0);
	//unsigned int j = get_global_id(1);

	if(which == 0){ //update A mat (reads from B mat)
    	if(i<N/2){
    		for (int j = 0; j<N; j++){
    		//this code could easily be halfed in length, but I didn't bother
            comp_ij =  
            (j%2==0)*B[id(i-1,j,N)] + (j%2==1)*B[id(i+1,j,N)] + //alternates top and bottom
            B[id(i,j-1,N)] + B[id(i,j+1,N)] +
            B[id(i,j,N)];
            comp_ij = A[id(i,j,N)]*comp_ij;
            
            tempVal = 0 + (comp_ij == 2)*comp_one + (comp_ij == 4)*comp_two;
            A[id(i,j,N)] = (-2*(tempVal == 0 || tempVal > rand[id(i,j,N)]) + 1) * A[id(i,j,N)]; 
    		}
    	}
    } else { //update B mat (reads from A mat)
        if(i<N/2){
    		for (int j = 0; j<N; j++){
    		
            comp_ij = 
            A[id(i,j,N)] +
            A[id(i,j-1,N)] + A[id(i,j+1,N)] +
            (j%2==0)*A[id(i+1,j,N)] + (j%2==1)*A[id(i-1,j,N)]; //alternates top and bottom
            comp_ij = B[id(i,j,N)]*comp_ij;
            
            tempVal = 0 + (comp_ij == 2)*comp_one + (comp_ij == 4)*comp_two;
            B[id(i,j,N)] = (-2*(tempVal == 0 || tempVal > rand[id(i,j,N)]) + 1) * B[id(i,j,N)]; 
    		}
        }
    }
}

__kernel void Magnetization //Find the Magnetization of a configuration. Likely no faster than np.sum
(__global int* A, __global int* B, const unsigned int N,  __global int* Mag)
{
    int comp = 0;
    unsigned int i = get_global_id(0);
    if(i<N*N/2){
        comp = comp+A[i]+B[i];
    Mag = comp;
    }
}

__kernel void Energy //Calculate the Energy of a Configuration
(__global int* A, __global int* B, __global int* Energy, const unsigned int N)
{
    int comp = 0;
    unsigned int i = get_global_id(0);
    //unsigned int j = get_global_id(1);
    
    if(i<N/2){
    	for (int j = 0; j<N; j++){
    	comp = comp + -1*A[id(i,j,N)] 
    	* (B[id(i,j+1,N)] + (j%2==0)*B[id(i,j,N)] + (j%2==1)*B[id(i+1,j,N)]);
    	comp = comp + -1*B[id(i,j,N)] 
    	* (A[id(i,j+1,N)] + (j%2==0)*A[id(i+1,j,N)] + (j%2==1)*A[id(i,j,N)]);	
    	}
    
    comp = comp/2;
    Energy = comp;
    //factor may be wrong, need to check
    }
}



