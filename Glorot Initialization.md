# Glorot Initialization

Glorot and He initializations are techniques used to mitigate the [[Vanishing and Exploding Gradients problem]] during the beginning of the training.

__Main Idea__ - We don't want the signal to die out (Vanishing Gradient), but we don't want it to explode and saturate either (Exploding Gradient).

As per Glorot et al., the variance of the outputs of any layer must be equal to the variance of the inputs of that layer to provide optimal results.  
   
#### Mic analogy
If you set a microphone's amplifier's knob too close to 0, the audience will not hear you, but if you set it too high, the voice will be saturated and people will not understand. Now if an amplifier is like a perceptron, we must tune the chain of the amplifiers correctly to get the signal through to the speakers.

### Initialization

Glorot initialization is done as:

$$ \text{Normal distribution with mean 0 and variance }\sigma^2 = \frac{1}{fan_{avg}} $$ 
$$ \text{Uniform distribution between} -r \text{ and } + r, \text { with } r = \sqrt{\frac{3}{fan_{avg}}}$$

where $fan_{avg} = \frac{fan_{in} + fan_{out}}{2}$
 
($fan$ is the number of neurons)

## LeCun
LeCun initialization is a variant of Glorot inilialization, when $fan_{in} = fan_{out}$. LeCun initialization, however, works best when used with the [[SELU]] activation function.

## He Initialization
The initlialization stratergy for [[ReLU]] activation function (and its variants) is called He Inilialization

<br>

| Initialization | Activation Functions | $\sigma^2$ (Normal) |
| ----------- | ----------- | -------- |
| Glorot | None, tanh, [[Sigmoid]], softmax | $\frac{1}{fan_{avg}}$ |
| He | [[ReLU]] and variants | $\frac{2}{fan_{in}}$ |
| LeCun | [[SELU]] | $\frac{1}{fan_{in}}$ |

<br>

<font color = red>__Note :__</font> By default, Keras uses glorot initialization

In order to change the initialization in Keras (to say He Initialization), you do the following: 

```python
keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')
```

or, if you want to use a uniform distribution with He initialization based on $fan_{avg}$ rather than $fan_{in}$, we can use `VarianceScaling`:

```python
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode3='fan_avg', distribution='uniform')
keras.layers.Dense(10, activation='sigmoid', kernal_initializer='he_avg_init')
```