# Echo-State-Network

## Project specification

This project implements a standard echo state network (ESN) model as discussed in class. This model does not include feedback from the output and do not use leaky-integrator neurons. Implement training of the read-out weights by means of ridge regression. Perform experiments and comment the results by considering a _k_ step ahead forecasting task on the "2sine" and "lorentz" time series data provided together with this notebook. Evaluate the impact of relevant hyper-parameters on the results, including the reservoir size and the amount of training data used for optimizing the read-out weights. Finally, discuss the effects of using different forecasting horizons on the overall performance of the model. **Note:** it is possible to consider the impact of additional hyperparameters, like the spectral radius of the reservoir matrix.

### K step ahead forecasting
A _k_ step ahead forecasting task consists of predicting the value of a time series at time $t+k$ by using the value of the time series at time $t$, where $k\geq1$ is called forecasting horizon.
In general, the predicted value is always unidimensional (i.e. a single number). However, it is possible to use multiple input values in order to improve the results. Notably, once _k_ is decided, the output to be predicted is the value of the time series at time $t+k$, and the input may be a vector containing values of the times series at time $t, t-1, \dots, t-n$, where $n\geq0$ is defined by the user and sets the dimensionality of the input vector.

### ESN Model With No Feedback

We define an ESN class that does not accept feedback.

The constructor of this ESN class accept:
- The reservoir size $N_r$
- A random seed (integer) to generate random numbers
- A scalar a that is used to scale the spectral radius
- A regularization coefficient reg_cof

#### The ESN model is defined as below:

$x_t = \phi(\boldsymbol{W^r} x_{t-1} + \boldsymbol{W^i} u_t)$


$z_t = \boldsymbol{W^o} x_t$


where x_t, u_t, z_t are states, inputs and output at discrete time step t respectively.

#### Parameters:

- $\boldsymbol{W^r}$: Recurrent layer weight matrix. 
  - $\boldsymbol{W^r}\in\mathbb{R}^{N_r * N_r}$, where $N_r$ is the number of neurons in the reservoir.

- $\boldsymbol{W^i}$: Input-to-reservoir weight matrix. 
  - $\boldsymbol{W^i}\in\mathbb{R}^{N_r * N_i}$, where $N_i$ is the input dimension.

- $\boldsymbol{W^o}$: read-output weight matrix.
  - $\boldsymbol{W^i}\in\mathbb{R}^{N_o * N_r}$, where $N_o$ is the output dimension.

When an ESN model instance is initialized, it does the following steps

- 1. Initialize model parameters $\boldsymbol{W^i}$ and $\boldsymbol{W^r}$ using the following rules:
  - Elements of $\boldsymbol{W^i}$ and $\boldsymbol{W^r}$ are indepndently drawn from a uniform distribution $[-1, 1]$
  - Find the spectral radius of $\boldsymbol{W^r}$, denoted as $p( \boldsymbol{W^r} )$. Then update $\boldsymbol{W^r}$ as: $\boldsymbol{W^r}$ = $a\frac{ \boldsymbol{W^r} }{ p( \boldsymbol{W^r} ) }$

- 2. Initialize the first state $x_0 = 0$
    
- 3. Feed the network with the sequence of input $u_i, i = 1, ..., N$


- 4. Collect the resulting N stages $x_i$ in a matrix $X \in \mathbb{R}^{N * N_r}$ 


- 5. Store all related targets in a $N-dimensional$ vector $t \in \mathbb{R}^{N}$


- 6. Find  $\boldsymbol{W^o}$ by solving the regularized least-square problem

### Regularized least-square problem for $W_o$

The problem is represented as: 

$\underbrace{arg\_min}_{\boldsymbol{W}\in\mathbb{R}^{N_r * N_o}}\frac{1}{2}\parallel XW - t\parallel^{2} + \frac{\lambda}{2}\parallel W\parallel^{2}$

which have closed form solution:

- $W = (X^{T}X + \lambda I)^{-1} X^{T}t$

#### MSE and MRSE formular
$MSE = <\parallel t - z \parallel^{2}>$, where <.> indicates average over time, t and z are target and predicted value respectively.

$NRMSE = \frac{\sqrt{MSE}}{var(t)}$
