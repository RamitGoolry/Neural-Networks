# Regression Neural Networks
Neural Networks can be trained for the task of [[Regression]].


### Output Layers
For __Univariate Regression__, we just have one output neuron.
For __Multivariate Regression__, we have multiple output neurons, equal to the output dimension.

Generally, we do not use any activation function for the output neuron, since we do not want to cap the range of our values.

However, we sometimes need to limit the range of our outputs, they can be done in the following way:

| Range of Output Values | Activation Function |
| --- | --- |
| $[0, \infty )$ | [[ReLU]], [[SoftPlus]] and variants |
| $[0, 1]$ | [[Sigmoid]] |
| $[-1, 1]$ | [[tanh]] |

### Loss
Typically, we use the [[Mean Squared Error]] as the loss function.