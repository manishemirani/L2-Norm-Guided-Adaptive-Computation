## L2 Norm Guided Adaptive Computation

This repo is the implementation of [L2 Norm Guided Adaptive Computation](https://openreview.net/forum?id=qW_GZYyn7C) in JAX.
Our paper introduces L2 Adaptive Computation (LAC), a new method that allows neural networks
to adapt their computational budget based per patch, example, or token level. LAC utilizes the
L2-norm of the model’s activations for making halting decisions, eliminating the need for extra
learnable parameters or auxiliary loss terms.

## Reference
If you use LAC(L2 Adaptive Computation), please cite the paper.
```
@InProceedings{shemiranifar2023l2norm,
    author    = {Shemiranifar, Mani and Dehghani, Mostafa},
    title     = {L2 Norm Guided Adaptive Computation},
    booktitle = {Proceedings of the Eleventh International Conference on Learning Representations (ICLR)},
    year      = {2023},
}
```