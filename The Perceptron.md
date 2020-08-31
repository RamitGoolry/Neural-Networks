# The Perceptron

The Perceptron is the fundamental building block of the [[Neural Network]], on top of which the whole architecture is built. They are supposed to simulate a biological neuron.

The perceptron does a simple mathematical operation - 
1. It computes a weighted sum of the inputs
2. Adds a bias term to the inputs
3. Applies a non-linear activation function.

![[Perceptron.png]]

Mathematically, the computation looks like:
$$h_{W, b}(X) = \phi(XW+b)$$

Where:
- $X$ is the input term
- $W$ is the weight matrix
- $b$ is the bias
- $\phi$ is the activation function

## The weight update

The weight update on a single layer is done as:

$$w_{i,j}^{\text{(next step)}} = w_{i,j} + \eta (y_j - \hat y_j) x_i$$

Where:
- $w_{i,j}$ is the connection between the $i^\text{th}$ input and the $j^\text{th}$ output.
- $x_i$ is the $i^\text{th}$ input.
- $\hat y_j$ is the $j^\text{th}$ output.
- $y_j$ is the $j^\text{th}$ target output.
- $\eta$ is the learning rate.

## Weakness
A single layer perceptron can only solve __linearly separable problems__, i.e., problems which can be separated by a line. Since most problems are not linearly separable, its falls short in this situation.

This kind of problem is more suited to a [[Neural Network|Multiplayer Perceptron]].