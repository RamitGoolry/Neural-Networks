# The Vanishing and Exploding Gradients problem
(a.k.a. The unstable gradients problem)

The Vanishing and Exploding Gradients problem comes up in [[Neural Network]] when using the [[Backpropogation Algorithm]]. 

When we are training a [[Neural Network]], the [[Backpropogation Algorithm]] works its way back in the network - from the output layer to the input layer. As a result, a lot of the times, large tweaks to the last few layers in the [[Neural Network]]s compensates for the loss and that ends up meaning that practically no change happens in the earlier layers. 

This is called the **vanishing gradients** problem. The oppositte effect takes place in [[Recurrent Neural Networks]], called the **exploding gradients** problem.

The vanishing gradients problem is an issue because even though the algorithm may converge on the solution, training time could have been faster, and there could have been much better feature extraction at the lower layers (which facilitates better [[Transfer Learning]]).

## Why does the Vanishing Gradients problem happen?
[Glorot](https://scholar.google.com/citations?user=_WnkXlkAAAAJ&hl=en) et al., in their [paper](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf?source=post_page---------------------------) found that the [[Sigmoid]] activation function might be a possible reason why the vanishing gradients problem comes about, along with the weight initialization scheme generally used (normal distribution with a mean of 0 and a standard deviation of 1).

Glorot found better success with the tanh activation function and the softsign activation function.


## Solutions to the problem: 
- [[Glorot Initialization]]
- [[Nonsaturating Activation Functions]]
- [[Batch Normalization]]
- [[Gradient Clipping]]