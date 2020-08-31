# Dying ReLU problem
We know that [[Gradient Descent]] works with the differential of the activation function, to adjust the neuron's weights. Now look at the Graph for the [[ReLU]] function. 

![[ReLU.png]]

When $x < 0$, $\frac{dy}{dx} = 0$. This means that when our neuron reaches a point, where it consistently output negative values, and since its gradient is 0, there is no way to fix the error of the function.

When a neuron can not produce an output it is called _dead_. In a normal training, about 50% of all neurons can die, especially if you are using a large learning rate.

In order to tackle the Dying ReLU problem, many new variants to [[ReLU]] were introduced, like [[Leaky ReLU]], [[ELU]] and [[SELU]], among others.

