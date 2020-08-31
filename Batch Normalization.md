# Batch Normalization

While [[Glorot Initialization]] can reduce the danger of the [[Vanishing and Exploding Gradients problem]], it does not completely remove it. It can come back during training. Batch Normalization tries to prevent this from happening.

## The Algorithm
The algorithm is applied right before or after the activation function of each hidden layer.

The Algorithm has 2 goals:
- Zero center and normalize the input
- Scale and shift the result using 2 parameter vectors layer.
	
It is performed in 4 steps:
1. __Calculation of Mean :__ 
	$$\mu_B = \frac{1}{m_B} \sum_{i=1}^{m_B} x^{(i)}$$
2. __Calculation of Standard Deviation:__
 $$\sigma_B^2 = \frac{1}{m_B} \sum_{i=1}^{m_B} (x^{(i)} - \mu_B)^2$$
3. __Calculation of the Normalized vector :__
$$\hat x^{(i)} = \frac{x^{(i) - \mu_B}}{\sqrt{\sigma_B^2 + \varepsilon}}$$
4. __Calculation of the rescaled and shifted input:__
$$z^{(i)} = \gamma \otimes \hat x^{(i)} + \beta$$

Here:
- $\mu_B$ is the vector of input means over a mini-batch $B$
- $\sigma_B$ is the vector of input standard deviations over a mini-batch $B$
- $m_B$ is the size of the mini-batch
- $\hat x^{(i)}$ is the zero-centered, normalized input
- $\gamma$ is the output scale vector
- $\otimes$ is element-wise multiplication
- $\beta$ is the output shift vector
- $\varepsilon$ is a tiny value to avoid division by zero
- $z^{(i)}$ is the output

The algorithm, however does not work as well in testing time, since sometimes we need the output of a single instance as compared to a batch. Therefore, the [[Neural Network]] is pretrained with the means and standard deviations of the whole training set.

#### Output parameters:
The algorithm learns 4 parameters:
- The output scale vector - $\gamma$
- The output offset vector - $\beta$
- The input mean vector - $\mu$
- The input standard deviation vector - $\sigma$

<font color = red> __NOTE :__ </font> Since Batch Normalization is basically standardizing the data and then scaling/shifting it, it can also be used as a [[Regularization]] technique.

## Merging with Dense Layer
Batch Normalization is a very expensive computation to make, and generally slows down the [[Neural Network]]. In order to get the same results without a compromise on speed, the [[Neural Network]] can fuse the `BatchNormalization` Layer with the preceeding hidden layer.

The goal is to take the hidden layer's output $XW + b$ and convert it to $\gamma \otimes \frac{XW + b - \mu}{\sigma} + \beta$.

We can define 2 new values $W'$ and $b'$ such that:
- $W' = \gamma \otimes \frac{W}{\sigma}$
- $b' = \gamma \otimes \frac{b - \mu}{\sigma} + \beta$

Then, $$XW' + B' \equiv \gamma \otimes \frac{XW + b - \mu}{\sigma} + \beta$$

## Usage

We can just add a `keras.layers.BatchNormalization()` layer between layers, as:

```python
model = keras.models.Sequential([
	keras.layers.Flatten(input_shape=[28, 28]),
	keras.layers.BatchNormalization(),     
	keras.layers.Dense(300, activation="elu",kernel_initializer="he_normal"),     
	keras.layers.BatchNormalization(),     
	keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),     
	keras.layers.BatchNormalization(),     
	keras.layers.Dense(10, activation="softmax") 
])

```
[TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)
It works better if you add the batch normalization laters before the Activation, as:

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
   	keras.layers.BatchNormalization(),
   	keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
   	keras.layers.BatchNormalization(),
   	keras.layers.Activation("elu"),
   	keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
   	keras.layers.BatchNormalization(),
   	keras.layers.Activation("elu"),
   	keras.layers.Dense(10, activation="softmax")
])
```

using `bias=False` allows for the layers to not use an activation function