#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=45G
#SBATCH --partition=gpu
#SBATCH --reservation=cpsc424gpu
#SBATCH -t 20:00
#SBATCH --job-name=flashattn
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
make flash_attention

# Now run the code. Note that if you turn on the error check using a
# cpu matmul code to check the answers, you will need more time for
# the job (possibly as much as 2 hours if you run all 4 test cases)
echo ""
echo "***Running Flash Attention module (n)"
# time ./flash_attention 128
# echo ""
# time ./flash_attention 256
# echo ""
# time ./flash_attention 512
# echo ""
# time ./flash_attention 1024 # GPT2
# echo ""
# time ./flash_attention 2048
# echo ""
# time ./flash_attention 4096
# echo ""
time ./flash_attention 8192
echo ""

echo "***All Done."