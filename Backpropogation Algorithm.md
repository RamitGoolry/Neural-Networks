# Backpropogation Algorithm

The backpropogation algorithm is used to propogate error through the neural network in order to tune the weights to learn complex patterns. It works as follows:

1. It handles one mini-batch at a time, each pass is called an _epoch_.
2. Each instance in the mini-batch is passed to the network's layers. The algorithm then computes the output in the output layer. This is called the _forward pass_.
3. Next, the algorithm measures the networks output error, based on some predefined loss function.
4. It computes how much each output connection has contributed to the error.
5. The algorithm then propogates the error throughout the network, using the [chain rule of derivatives](https://en.wikipedia.org/wiki/Chain_rule).
6. Finally, the algorithm performs a [[Gradient Descent]] step to tweak all the weights in the network, in order to improve the loss.

### Technical Clarification
1. It is important that all the weights in the layers are initialized randomly. If all the weights are the same, all the changes from the [[Gradient Descent]] step will remain identical, and as a result the network will act as though there was only _one neuron per layer_.
2. We can not use the [[Step Function]] with the Backpropogation algorithm as an activation function. It does not have a derivative at every point, making backpropogation fail in these cases. Instead, a [[Sigmoid|Sigmoid Curve]] is used. 