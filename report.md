# CS217 Final Project report

**Team name**: VOID

**Group member**: Xingyan Zhou, Xinyu Zhang and Zhaorui Yang

**Project option**: Parallelizing serial C code with CUDA

# Project idea

The Backpropagation Network source code is a C language implementation of a neural network simulator, focusing on the backpropagation algorithm. It's primarily used for time-series forecasting, such as predicting the annual number of sunspots. The code includes the definition of the network structure, random number generation functions, and the core algorithms for learning and prediction.

Our team has thoroughly reviewed the source code and found that the backpropagation network consists of both forward propagation and backpropagation. As a result, we can parallelize the code in two directions. Additionally, we identified several matrix and vector multiplications that we had implemented in our assignments, we decided to integrated them into the project.

Furthermore, we're consider to implement a timing mechanism within the source code. This feature will help to compare the performance metrics before and after modifications.

# Our works
### Implementation details

**backpropagate.cu**
```C

```

**bpn.c**
```C

```

**main.c**
```C

```

**main.cu**
```C

```

**propagate.cu**
```C

```

**utils.cu**
```C

```

### How to run



# Results



# Conclusion



# Contribution
