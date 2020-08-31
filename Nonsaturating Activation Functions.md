# Nonsaturating Activation Functions
The [[Vanishing and Exploding Gradients problem]] is partly because of poor choice of activation function. The [[Sigmoid]] function was used, since it is similar to what is found in natural neurons. However, the [[ReLU]] function has shown better promise, since it does not saturate.

The only exception to this statement is when you are using [[Sigmoid]] for the output layer of a binary classifier, because it sets the output in the range $[0, 1]$, which ReLU does not.

## ReLU is not perfect, however. 
ReLU neurons suffer from a problem known as the [[Dying ReLU]] problem. This is fixed by introducing different variants of the ReLU function.
As a result, many other functions, such as [[Leaky ReLU]], [[ELU]] and [[SELU]] have been introduced.

