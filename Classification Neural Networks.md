# Classification Neural Networks

[[Neural Network|Neural Networks]] can also be used for [[Classification]] tasks.

### Output Layers
For a __Binary Classification__ task, we only need a single output neuron using the [[Sigmoid]] activation function.

For a __Multilabel Classification__ task (if an instance belongs to a single class out of multiple), we can use an array of output neurons, for which the data is one-hot encoded. We use the [[Softmax]] activation function.