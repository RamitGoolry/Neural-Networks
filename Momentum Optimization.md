# Momentum Optimization

One of the drawbacks of [[Gradient Descent]] is that it does not care about the previous gradients, only the local gradient. On the contrary, Momentum optimization considers the previous gradients. 

Think about optimization as a ball going down a slope. 

![[Gradient Descent Slope.jpg]]

> "In momentum optimization, the gradient is used for acceleration, not for speed"

To simulate friction, such that the momentum does not grow too large, another hyperparameter $\beta$ is added, which is between 0 (high friction) to 1 (no friction). 

Momentum is generally around 0.9.

## The Algorithm
The momentum optimization algorithm has 2 steps:

1. $$m \leftarrow \beta m - \eta \nabla_{\theta} J(\theta) $$
2. $$ \theta \leftarrow \theta + m $$

## Usage 
Momentum optimization in Keras just is used as a parameter of the `SGD` optimizer.

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum = 0.9)
```