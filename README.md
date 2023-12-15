# flash-attention-cuda

CPSC 524 Final Project.

This is a CUDA c implementation of the [Flash Attention](https://arxiv.org/abs/2205.14135) paper. Specifically I focus on forward pass of the attention mechanism without multi-head attention. This is a work in progress and I will be adding more features in the future.

### For now, I have implemented the following:
- [x] CPU implementation of the attention mechanism
- [x] GPU naive implementation of the attention mechanism
- [x] Forward pass of Flash Attention without multi-head attention

### To do (outside of the scope of this project):
- [ ] Backward pass of Flash Attention without multi-head attention
- [ ] Multi-head attention
- [ ] Options for masking, dropout, etc.
- [ ] Integration with PyTorch

### Run scripts
(Each GPU attention implementation will be compared against the CPU implementation for error checking, you can comment out the CPU implementation if you don't want to run it)

```bash
sbatch run-standard.sh # naive GPU implementation
```

```bash
sbatch run-flash.sh # forward flash attention
```

### Results







