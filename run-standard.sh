#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=45G
#SBATCH --partition=gpu
#SBATCH --reservation=cpsc424gpu
#SBATCH -t 20:00
#SBATCH --job-name=attn
#SBATCH --gpus=1

echo "***Purging module files"
echo ""
module purge
echo ""
echo "***Loading CUDA module file"
echo ""
module load CUDA
echo ""
module list

echo ""
echo "***Running nvidia-smi"
echo ""
nvidia-smi
echo ""
echo ""

echo "***Running deviceQuery"
/vast/palmer/apps/avx.grace/software/CUDAcore/11.3.1/extras/demo_suite/deviceQuery
echo ""

echo "***Building matmul"
make clean
make attn

# Now run the code. Note that if you turn on the error check using a
# cpu matmul code to check the answers, you will need more time for
# the job (possibly as much as 2 hours if you run all 4 test cases)
echo ""
echo "***Running Standard Attention module (n, d, block dim)"
time ./attn 128 64 16
echo ""
time ./attn 128 64 16
echo ""
time ./attn 1024 64 16 # GPT2
echo ""


echo "***All Done."