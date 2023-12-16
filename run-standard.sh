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
# echo ""
# echo "***Running Standard Attention module (n, d, block dim)"
# time ./attn 128 64 16
# echo ""
# time ./attn 256 64 16
# echo ""
# time ./attn 512 64 16
echo ""
time ./attn 1024 64 16 # GPT2
echo ""
time ./attn 2048 64 16 # GPT2
echo ""
time ./attn 4096 64 16 # GPT2
echo ""
time ./attn 8192 64 16 # GPT2
echo ""

# # Fake MHA 16 heads
# echo "***Simulating MHA 16 heads, Standard Attention module (n, d, block dim)"
# time ./attn 1024 4 16 # GPT2
# echo ""
# time ./attn 2048 4 16 # GPT3
# echo ""
# time ./attn 4096 4 16 # GPT2
# echo ""
# time ./attn 8192 4 16 # GPT3
# echo ""

echo "***Running Standard Attention module (n, d, block dim)"
# time ./attn 128 1024 16
# echo ""
# time ./attn 256 1024 16
# echo ""
# time ./attn 512 1024 16
# echo ""
# time ./attn 1024 1024 16 # GPT2
# echo ""
# time ./attn 2048 1024 16 # GPT2
# echo ""
# time ./attn 4096 1024 16 # GPT2
# echo ""
# time ./attn 8192 1024 16 # GPT2

# time ./attn 128 32 16
# echo ""
# time ./attn 256 32 16
# echo ""
# time ./attn 512 32 16
# echo ""
# time ./attn 1024 32 16 # GPT2
# echo ""
# time ./attn 2048 32 16 # GPT2
# echo ""
# time ./attn 4096 32 16 # GPT2
# echo ""
# time ./attn 8192 32 16 # GPT2

# time ./attn 128 16 16
# echo ""
# time ./attn 256 16 16
# echo ""
# time ./attn 512 16 16
# echo ""
# time ./attn 1024 16 16 # GPT2
# echo ""
# time ./attn 2048 16 16 # GPT2
# echo ""
# time ./attn 4096 16 16 # GPT2
# echo ""
# time ./attn 8192 16 16 # GPT2
# echo ""

# time ./attn 128 8 16
# echo ""
# time ./attn 256 8 16
# echo ""
# time ./attn 512 8 16
# echo ""
# time ./attn 1024 8 16 # GPT2
# echo ""
# time ./attn 2048 8 16 # GPT2
# echo ""
# time ./attn 4096 8 16 # GPT2
# echo ""
# time ./attn 8192 8 16 # GPT2
# echo ""

echo "***All Done."