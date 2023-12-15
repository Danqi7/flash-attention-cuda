#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>


// Fused flash attention kernel
void flash_attn(FP *Q, FP *K, FP *V, FP *O, FP *l, FP *m, int n, int d, int Br, int Bc) {
    /*
        Q, V, K, O : [n, d]; l, m : [n,]
        Divide Q into Q1, Q2, ..., QTr of shape [Br, d]
        Divide K, V in K1/V1, ..., KTc/VTc of shape [Bc, d]
        Divide O into O1, O2, ..., OTr of shape [Br, d]
        Divide l, m into l1/m1, ..., lTr/mTr of shape [Br,]
    */

    int Tr = (n + Br - 1) / Br;
    int Tc = (n + Bc - 1) / Bc;
    
    __shared__ FP Kj[Bc][d];
    __shared__ FP Vj[Bc][d];
    __shared__ FP Qi[Br][d];
    __shared__ FP Oi[Br][d];
    __shared__ FP Sij[Br][Bc];
    
    __shared__ FP tmp_Oi[Br][d];

    __shared__ FP li[Br];
    __shared__ FP mi[Br];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = tx + blockDim.x * blockIdx.x; // col is tx when GridDim.x = 1
    int row = ty + blockDim.y * blockIdx.y;

    for (int j = 0; j < Tc; j++) {
        // Load Kj, Vj from HBM to on-chip SRAM
        if (col < n) {
            for (int idx=0; idx < d: idx++) {
                //Kj[tx][idx] = K[col][idx]
                Kj[tx*d + idx] = K[j*Bc + tx*d + idx];
                Vj[tx*d + idx] = V[j*Bc + tx*d + idx];
            }
        }
        
        // Load Qi, Oi, li, mi from HBM to on-chip SRAM
        if (row < n) {
            for (int idx=0; idx < d: idx++) {
                //Q[ty][idx] = Q[row*d][idx]
                Qi[ty*d + idx] = Q[row * d + idx];
                Oi[ty*d + idx] = O[row * d + idx];
            }
            li[ty] = l[row];
            mi[ty] = m[row];
        }

        // make sure Qi, Oi, Kj, Vj are loaded
        __syncthreads();

        // Compute Sij = QiKj^T
        FP sVal = 0.0;
        for (int idx=0; idx < d: idx++) {
            sVal += Qj[ty*d + idx] * Kj[tx*d + idx];
        }
        Sij[ty*Bc + tx] = sVal; // Sij is of size [Br x Bc]
        // make sure Sij is computed
        __syncthreads();

        // Compute rowmax m^~_ij
        // mij_delta[ty] = max(mij_delta[ty], Sij[ty*Bc + tx]); // Assume mij_delta is initialized to -inf
        // __syncthreads();
        FP mij_delta = -inf;
        for (int idx=0; idx < Bc; idx++) {
            mij_delta = max(mij_delta, Sij[ty * Bc + idx]);
        }
        // P^~_ij
        //Pij_delta = exp(S[ty*Bc + tx] - m_delta[ty]);
        FP Pij_delta = exp(Sij[ty*Bc + tx] - mij_delta);

        // l^~_ij
        //lij_delta[ty] += Pij_delta; // Assume l_delta is initialized to 0
        //__syncthreads();
        FP lij_delta = 0;
        for (int idx=0; idx < Bc; idx++) {
            lij_delta += Pij_delta[ty * Bc + idx];
        }

        // Compute new m, l
        FP mi_new = max(mi[ty], mij_delta);
        FP coeffold = exp(mi[ty] - mi_new);
        FP coeffcur = exp(mij_delta - mi_new);
        li_new = coeffold * li[ty] + coeffcur * lij_delta; // per row in Br

        // Update Oi: each row doing ([1,Bc]).dot([Bc,d]) = [1,d]
        for (int idx=0; idx < d: idx++) {
            tmp_Oi[ty*d + idx] = 0;
        }
        for (int idx=0; idx < d: idx++) {
            tmp_Oi[ty*d + idx] += (li[ty] * coeffold * Oi[ty*d+idx] + coeffcur * Pij_delta * Vj[tx*d+idx]) / li_new;
        }
        __syncthreads();

        // Update Oi in HBM
        if (row < n) {
            for (int idx=0; idx < d: idx++) {
                O[row*d + idx] = tmp_Oi[ty*d + idx];
            }
        }

        // Update li, mi in HBM
        li[ty] = li_new;
        mi[ty] = mi_new;
        __syncthreads();

    }   
}

int main() {
  int i, j; // loop counters

  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use
  int Grid_Dim_X = 1; //Grid x dimension
  int Grid_Dim_Y = 1; // Grid y dim
  int Block_Dim_X = 1; //Block dimension
  int Block_Dim_Y = 1;

  int n, d;
  FP *q,*k,*v, *o, *p; // Q,K,V,O are (n, d) inputs, S,P are intermediate (n, n) matrix
  FP *dev_q, *dev_k, *dev_v, *dev_s, *dev_p, *dev_o;
  size_t Qsize, Ksize, Vsize, Ssize, Psize, Osize; // number of bytes in arrays
  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  // --------------------SET PARAMETERS AND DATA -----------------------

  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else {
     printf("Device count = %d\n",gpucount);
  }

  if (argc!=5) {
    printf("Usage: flash_attn <matrix dim n> <matrix dim d> <block dim Y Br> <block dim X Bc>\n");
    exit (-1);
  }

  n = atoi(argv[1]);
  d = atoi(argv[2]);
  Block_Dim_Y = atoi(argv[3]); // Non-square block, [Br, Bc]
  Block_Dim_X = atoi(argv[4]);
  if (Block_Dim_Y*Block_Dim_X > 1024) {
    printf("Error, too many threads in block\n");
    exit (-1);
  }

  cudaSetDevice(gpunum);
  printf("Using device %d\n",gpunum);
  printf("Matrix Q,K,V Dimension = [%d, %d]\n", n, d);

  Qsize = n * d * sizeof(FP);
  Ksize = n * d * sizeof(FP);
  Vsize = n * d * sizeof(FP);
  Osize = n * d * sizeof(FP);

  q = (FP*) malloc(Qsize); // dynamically allocated memory for arrays on host
  k = (FP*) malloc(Ksize);
  v = (FP*) malloc(Vsize);
  o = (FP*) malloc(Osize); // final output

  srand(12345);
  for(i=0;i < n;i++)
    for(j=0;j < d;j++) {
      q[i * d + j] = (FP) rand() / (FP) RAND_MAX;
      //      a[i * p + j] = (FP) i+j; // may be helpful for debugging
    }
  for(i=0;i < n;i++)
    for(j=0;j < d;j++) {
      k[i * d + j] = (FP) rand() / (FP) RAND_MAX;
      //      b[i * n + j] = (FP) i+j; // may be helpful for debugging
    }
  for(i=0;i < n;i++)
    for(j=0;j < d;j++) {
      v[i * d + j] = (FP) rand() / (FP) RAND_MAX;
      //      b[i * n + j] = (FP) i+j; // may be helpful for debugging
    }

  // ------------- COMPUTATION DONE ON GPU ----------------------------

  cudaMalloc((void**)&dev_q, Qsize); // allocate memory on device
  cudaMalloc((void**)&dev_k, Ksize);
  cudaMalloc((void**)&dev_v, Vsize);
  cudaMalloc((void**)&dev_o, Osize);
  cudaMalloc((void**)&dev_l, n * sizeof(FP));
  cudaMalloc((void**)&dev_m, n * sizeof(FP));

  cudaMemcpy(dev_q, q , Qsize ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_k, k , Ksize ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v, v , Vsize ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_o, o , Osize ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_l, l , n * sizeof(FP) ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_m, m , n * sizeof(FP) ,cudaMemcpyHostToDevice);

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);
  // cudaEventSynchronize(start); // not needed

  Grid_Dim_X = 1; //  how many blocks in X direction
  Grid_Dim_Y = (n + Block_Dim_Y - 1) / Block_Dim_Y; // how many blocks in Y direction
  if (Grid_Dim_X*Block_Dim_x < n) {
    printf("Error, number of threads in x dimensions less than number of array elements\n");
    exit (-1);
  }
  if (Grid_Dim_Y*Block_Dim_Y < n) {
    printf("QK^T: Error, number of threads in y dimensions less than number of array elements\n");
    exit (-1);
  }
  printf("Block_Dim = (%d, %d), Grid_Dim = (%d, %d)\n", Block_Dim_X, Block_Dim_Y, Grid_Dim_X, Grid_Dim_Y);
  dim3 Grid(Grid_Dim_X, Grid_Dim_Y); //Grid structure
  dim3 Block(Block_Dim_X, Block_Dim_Y); //Block structure
  
  flash_attn<<<Grid,Block>>>(dev_q, dev_k, dev_v, dev_o, n, d, Block_Dim_Y, Block_Dim_X);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  printf("Time for Flash Attention on GPU: %f ms.\n", elapsed_time_ms);


// START OF OPTIONAL SECTION THAT CAN BE OMITTED

  // ------------- COMPUTATION DONE ON HOST CPU ----------------------------
  // DEBUGGING USE ONLY (AND FOR LIMITED NUMBERS OF TIMING RUNS)

  cudaEventRecord(start, 0); // use same timing
  // cudaEventSynchronize(start); // not needed

  //cpu_matrixmult(a,b,c, n, p, m); // do calculation on host (NOTE: This computes the diff with GPU result.)
  p = (FP*) malloc(Psize);
  cpu_attention(q,k,v, p,o, n,d);

  cudaEventRecord(stop, 0); // instrument code to measue end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );

  printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms); // exec. time

// ------------------- check device creates correct results -----------------

  double error, sumv, vi;
  sumv = 0.;
  for(i=0;i < n*d; i++) {
    vi = (double) v[i];
    sumv += vi * vi;
  }
  sumv = sqrt(sumv);
  error =  sumv / (n*d);
  printf("Approximate relative error between GPU and CPU: %e\n", error);
  free(p);
// END OF OPTIONAL SECTION THAT CAN BE OMITTED
// -------------- clean up ---------------------------------------

  free(q);
  free(k);
  free(v);
  free(o);
  cudaFree(dev_q);
  cudaFree(dev_k);
  cudaFree(dev_v);
  cudaFree(dev_o);
  cudaFree(dev_l);
  cudaFree(dev_m);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
