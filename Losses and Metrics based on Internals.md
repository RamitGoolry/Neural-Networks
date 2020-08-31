# Losses and Metrics based on Internals

Sometimes, it is useful to evaluate a properto of a model based on an internal state of its, such as the weights and activations of its hidden layers.

An example of this is [[Reconstruction Loss]], something that will work very will in a network like an [[Autoencoder]].

A custom loss can be added to the model using the `add_loss()` method. This method adds a different metric to the loss function, which is trying to be minimized.

In the example of [[Reconstruction Loss]], the network will try to minimize it, which will allow the network to reach a closer value to the input, which why it is very helpful for [[Autoencoder|Autoencoders]].

## Example :
```python
class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                   kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
		
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs) # Extra dense layer to reconstruct the inputs
        super().build(batch_input_shape)
		
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```