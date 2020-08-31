# Nesterov Accelerated Gradient
(a.k.a. Nesterov momentum optimization)

Nesterove Accelerated Gradient does the exact same thing that [[Momentum Optimization]] does, only it does not measure the gradient at the local position $\theta$ but peeks slightly ahead to $\theta + \beta m$. 

Think about it, when you are driving, you would want to turn slightly before the turn actually comes, so that the momentum already in the car is balenced out.

The Nesterov update ends up working better than standard Momentum Optimization.

## The Algorithm
1. $$m \leftarrow \beta m - \eta \nabla_{\theta} J(\theta + \beta m)$$
2. $$\theta \leftarrow \theta + m$$

## Usage
We set `nesterov = True` when using the `SGD` optimizer

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
```