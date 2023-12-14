#define FP float

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

#define S_MAX 1000000

__global__ void gpu_qk_matmul(FP *q,FP *k, FP *s, int n, int d) {
  // Q: [n, d], K: [n, d], S[n, n]; usually n>>d
  // S = QK^T
  // Naive version: each thread is computing one element of S

  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  int indexk = col * d; // transpose
  int index = row * n + col;
  
  if(col < n && row < n) {
    s[index] = 0.;
    for (int indexq = row*d; indexq < (row*d + d); indexq++, indexk++) 
      s[index] += q[indexq]*k[indexk];
  }
  
}

__global__ void gpu_pv_matmul(FP *p,FP *v, FP *o, int n, int d) {
  // P: [n, n], V: [n, d], O[n, d]; usually n>>d
  // O = PV, sum(P) rowwise need to be 1.
  // Naive version: each thread is computing one element of O

  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  int indexv = col;
  int index = row * d + col;
  
  if(col < d && row < n) {
    o[index] = 0.;
    for (int indexp = row*n; indexp < (row*n + n); indexp++, indexv+=d) 
      o[index] += p[indexp]*v[indexv];
  }
  
}

__global__ void gpu_softmax(FP *s, FP *p, int n) {
  // P = softmax(S) row-wise
  // S[n, n], P[n, n]
  // Naive version: each thread is computing whole row of P

  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if (row < n) {
    // subtract max of row for numerical stability
    FP rowMax = -S_MAX;
    for (int i = 0; i < n; i++) {
        rowMax = max(rowMax, s[row * n + i]);
    }

    // sum up elements in the same row
    // sum_i^n e^(s_row_i - rowMax)
    FP rowSum = 0.0;
    for (int i = 0; i < n; i++) {
      rowSum += exp(s[row * n + i] - rowMax);
    }

    // P := e^{row_i} / sum_j^n e^{row_j}
    int index;
    for (int i = 0; i < n; i++) {
      index = row * n + i;
      p[index] = exp(s[index] - rowMax) / rowSum;
    }
  }

}

void cpu_attention(FP *q,FP *k, FP *v, FP *o, FP *p, int n, int d) {
  // Q, K, V, O: [n, d]. usually n>>d
  FP rowMax, rowSum;
  int indexq, indexk;
  for (int row=0; row<n; row++) {
    rowMax = S_MAX;
    rowSum = 0.0;

    for (int col=0; col<n; col++) {
      sVal = 0;
      indexk = col*d;
      for (indexq = row*d; indexq < (row*d + d); indexq++, indexk++) {
        sVal += q[indexq] * k[indexk];
      }
      rowMax = max(rowMax, sVal);
      rowSum += sVal;
      p[row*n+col] = sVal; // unnormalized
    }

    // Normalize for each row
    for (int col=0; col < n; col++)
      p[row*n+col] = (p[row*n+col] - rowMax) / rowSum;
  }


}

void cpu_matrixmult(FP *a,FP *b, FP *c, int n, int p, int m) {
  // A: [n,p], B: [p, m], C: [n, m]
  int index, indexa, indexb;
  FP cvalue;
  for(int col=0;col < m; col++)
    for(int row=0;row < n; row++) {
      indexb = col;
      index = row * m + col;
      cvalue = 0.;
      for (indexa = row*p; indexa < (row*p + p); indexa++, indexb+=m) 
	      cvalue += a[indexa]*b[indexb];
      c[index] -= cvalue; //NOTE: This calculates the diff between CPU and GPU computations.
    }
}

void cpu_matrixmult_kij(FP *a,FP *b, FP *c, int n, int p, int m) {
  // A: [n,p], B: [p, m], C: [n, m]
  int index, indexa, indexb;
  FP r;
  
  for (int k=0; k<p; k++) {

    for (int i=0; i<n; i++) {
      indexa = i * p + k;
      // r = A[i][k];
      r = a[indexa];
      for (int j=0; j<m; j++) {
        index = i * m + j;
        indexb = k * m + j;
        //C[i][j] += r * B[k][j];
        c[index] = c[index] - (r * b[indexb]); // NOTE: diff between CPU and GPU computations.
      }
    }

  }
}

int main(int argc, char *argv[]) {

  int i, j; // loop counters

  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use
  int Grid_Dim_X = 1; //Grid x dimension
  int Grid_Dim_Y = 1; // Grid y dim
  int Block_Dim = 1; //Block dimension, x and y, square

  int n, d;
  FP *q,*k,*v, *o; // Q,K,V,O are (n, d) inputs, S,P are intermediate (n, n) matrix
  FP *dev_q, *dev_k, *dev_v, *dev_s, *dev_p, *dev_o;
  size_t Qsize, Ksize, Vsize, Ssize, Psize, Osize; // number of bytes in arrays
  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  float total_elapsed = 0;
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

  if (argc!=4) {
    printf("Usage: attn <matrix dim n> <matrix dim d> <block dim>\n");
    exit (-1);
  }

  n = atoi(argv[1]);
  d = atoi(argv[2]);
  Block_Dim = atoi(argv[3]); // Square block
  if (Block_Dim*Block_Dim > 1024) {
    printf("Error, too many threads in block\n");
    exit (-1);
  }

  cudaSetDevice(gpunum);
  printf("Using device %d\n",gpunum);
  printf("Matrix Q,K,V Dimension = [%d, %d]\n", n, d);

  Qsize = n * d * sizeof(FP);
  Ksize = n * d * sizeof(FP);
  Vsize = n * d * sizeof(FP);
  Ssize = n * n * sizeof(FP);
  Psize = n * n * sizeof(FP);
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
  cudaMalloc((void**)&dev_s, Ssize);
  
  cudaMemcpy(dev_q, q , Qsize ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_k, k , Ksize ,cudaMemcpyHostToDevice);

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);
  // cudaEventSynchronize(start); // not needed
  // Step 1: S = QK^T
  Grid_Dim_X = (n + Block_Dim - 1) / Block_Dim; // rectangular grid; how many blocks in X direction
  Grid_Dim_Y = (n + Block_Dim - 1) / Block_Dim; // how many blocks in Y direction
  if (Grid_Dim_X*Block_Dim < n) {
    printf("Error, number of threads in x dimensions less than number of array elements\n");
    exit (-1);
  }
  if (Grid_Dim_Y*Block_Dim < n) {
    printf("QK^T: Error, number of threads in y dimensions less than number of array elements\n");
    exit (-1);
  }
  printf("QK^T Block_Dim = (%d, %d), Grid_Dim = (%d, %d)\n", Block_Dim, Block_Dim, Grid_Dim_X, Grid_Dim_Y);
  dim3 Grid(Grid_Dim_X, Grid_Dim_Y); //Grid structure
  dim3 Block(Block_Dim, Block_Dim); //Block structure
  
  //gpu_matrixmult<<<Grid,Block>>>(dev_a,dev_b,dev_c, n, p, m);
  gpu_qk_matmul<<<Grid,Block>>>(dev_q, dev_k, dev_s, n, d);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  //cudaMemcpy(s, dev_s, Ssize ,cudaMemcpyDeviceToHost); no need 
  printf("Time to calculate QK^T on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  total_elapsed += elapsed_time_ms;

  // Step 2: P = softmax(S)
  cudaMalloc((void**)&dev_s, Ssize);
  cudaMalloc((void**)&dev_p, Psize);

  if ((n + Block_Dim - 1)/Block_Dim *Block_Dim < n) {
    printf("Softmax: Error, number of threads in y dimensions less than number of array elements\n");
    exit (-1);
  }
  printf("Softmax Block_Dim = (%d, %d), Grid_Dim = (%d, %d)\n", 1, Block_Dim, 1, (n + Block_Dim - 1)/Block_Dim);

  dim3 SoftmaxGrid(1, (n + Block_Dim - 1)/Block_Dim); //Grid structure
  dim3 SoftmaxBlock(1, Block_Dim); //Block structure
  cudaEventRecord(start, 0); // use same timing
  gpu_softmax<<<SoftmaxGrid, SoftmaxBlock>>>(dev_s, dev_p, n);
  cudaEventRecord(stop, 0); // instrument code to measue end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );
  printf("Time to calculate softmax(S) on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  total_elapsed += elapsed_time_ms;

  // Step 3: O=PV
  cudaMalloc((void**)&dev_o, Osize);
  cudaMalloc((void**)&dev_v, Vsize);
  cudaMemcpy(dev_v, v, Vsize, cudaMemcpyHostToDevice);

  Grid_Dim_X = (d + Block_Dim - 1) / Block_Dim; // rectangular grid; how many blocks in X direction
  Grid_Dim_Y = (n + Block_Dim - 1) / Block_Dim; // how many blocks in Y direction
  if (Grid_Dim_X*Block_Dim < d) {
    printf("Error, number of threads in x dimensions less than number of array elements\n");
    exit (-1);
  }
  if (Grid_Dim_Y*Block_Dim < n) {
    printf("QK^T: Error, number of threads in y dimensions less than number of array elements\n");
    exit (-1);
  }
  printf("QK^T Block_Dim = (%d, %d), Grid_Dim = (%d, %d)\n", Block_Dim, Block_Dim, Grid_Dim_X, Grid_Dim_Y);
  dim3 OutGrid(Grid_Dim_X, Grid_Dim_Y); //Grid structure
  dim3 OutBlock(Block_Dim, Block_Dim); //Block structure
  
  cudaEventRecord(start, 0);
  gpu_pv_matmul<<<OutGrid,OutBlock>>>(dev_p, dev_v, dev_o, n, d);
  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaMemcpy(o, dev_o, Osize ,cudaMemcpyDeviceToHost); // copy output to CPU
  printf("Time to calculate PV on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  total_elapsed += elapsed_time_ms;

  printf("Total Time for standard Attention on GPU: %f ms.\n", total_elapsed);


// START OF OPTIONAL SECTION THAT CAN BE OMITTED
/*
  // ------------- COMPUTATION DONE ON HOST CPU ----------------------------
  // DEBUGGING USE ONLY (AND FOR LIMITED NUMBERS OF TIMING RUNS)

  cudaEventRecord(start, 0); // use same timing
  // cudaEventSynchronize(start); // not needed


  //cpu_matrixmult(a,b,c, n, p, m); // do calculation on host (NOTE: This computes the diff with GPU result.)
  cpu_matrixmult_kij(a,b,c, n, p, m); // do calculation on host (NOTE: This computes the diff with GPU result.)

  cudaEventRecord(stop, 0); // instrument code to measue end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );

  printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms); // exec. time

// ------------------- check device creates correct results -----------------

  double error, suma, sumb, sumc, ai, bi, ci;
  suma = 0.; sumb = 0; sumc = 0;
  for(i=0;i < n*n;i++) {
    ai = (double) a[i];
    bi = (double) b[i];
    ci = (double) c[i];
    suma += ai*ai;
    sumb += bi*bi;
    sumc += ci*ci;
  }
  suma = sqrt(suma);
  sumb = sqrt(sumb);
  sumc = sqrt(sumc);
  error =  sumc/(suma*sumb);
  printf("Approximate relative error between GPU and CPU: %e\n", error);
*/
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
  cudaFree(dev_s);
  cudaFree(dev_p);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
