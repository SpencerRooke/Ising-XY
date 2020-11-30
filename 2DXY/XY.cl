#define id(x,y,N) ( ((x)%(N/2))*N + ((y)%N) )
# define pi 3.14159265
//this can definitely be implemented better, will revisit when better with open cl
__kernel void Comparison
(__global float* A, __global float* B, __global float* rand, __global float* dPhi, 
float Tau, const unsigned int N,int which)
{
    float J = .25;
    float E_ij;
    float E_n;
    float tempVal;
	unsigned int i = get_global_id(0);
	//unsigned int j = get_global_id(1);

	if(which == 0){ //update A mat (reads from B mat)
    	if(i<N/2){
    		for (int j = 0; j<N; j++){
    		//this code could easily be quartered in length
    		float cur_phi = A[id(i,j,N)];
            E_ij =   //alternates top and bottom
            cos(cur_phi-((j%2==0)*B[id(i-1,j,N)] + (j%2==1)*B[id(i+1,j,N)]) ) +
            cos(cur_phi-B[id(i,j-1,N)]) +
            cos(cur_phi-B[id(i,j+1,N)]) +
            cos(cur_phi-B[id(i,j,N)]);
            
            float deltaP = dPhi[id(i,j,N)];
            float comp_phi = cur_phi+deltaP;
            E_n = 
            cos(comp_phi-((j%2==0)*B[id(i-1,j,N)] + (j%2==1)*B[id(i+1,j,N)]) ) +
            cos(comp_phi-B[id(i,j-1,N)]) +
            cos(comp_phi-B[id(i,j+1,N)]) +
            cos(comp_phi-B[id(i,j,N)]);
            
            A[id(i,j,N)] += deltaP *(exp( (E_n-E_ij) * (-2*J/Tau)) > rand[id(i,j,N)]);
            A[id(i,j,N)] += (A[id(i,j,N)] < -pi) * pi + (A[id(i,j,N)] > pi) * (-pi);
    		}
    	}
    } else { //update B mat (reads from A mat)
        if(i<N/2){
    		for (int j = 0; j<N; j++){
    		float cur_phi = B[id(i,j,N)];
            E_ij = 
            cos(cur_phi-A[id(i,j,N)]) +
            cos(cur_phi-A[id(i,j-1,N)]) +
            cos(cur_phi-A[id(i,j+1,N)]) +
            cos(cur_phi-((j%2==0)*A[id(i+1,j,N)] + (j%2==1)*A[id(i-1,j,N)]) ); 
            //alternates top and bottom
            
            float deltaP = dPhi[id(i,j,N)];
            float comp_phi = cur_phi+deltaP;
            E_n = 
            cos(comp_phi-A[id(i,j,N)]) +
            cos(comp_phi-A[id(i,j-1,N)]) +
            cos(comp_phi-A[id(i,j+1,N)]) +
            cos(comp_phi-((j%2==0)*A[id(i+1,j,N)] + (j%2==1)*A[id(i-1,j,N)]) );
            
            B[id(i,j,N)] += deltaP *(exp( (E_n-E_ij) * (-2*J/Tau)) > rand[id(i,j,N)]);
            B[id(i,j,N)] += (B[id(i,j,N)] < -pi) * pi + (B[id(i,j,N)] > pi) * (-pi);
    		}
        }
    }
}



