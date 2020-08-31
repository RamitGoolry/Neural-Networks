# Transfer Learning
Transfer Learning is another method to speed up training in [[Neural Network]]. 

The idea behind transfer learning is to reuse the lower layers of another model that has already been trained for a similar task, instead of starting training from scratch.

This way, it __speeds up training considerably__ and also __requires less training data__.

#### Why can we only use the lower layers of a [[Neural Network]]?
The upper hidden layers are less likely to be useful as the lower layers. This is because the upper hidden layers are generally more specific to the task, while the lower layers are more pertinent to the data. 
The lower levels focus on __feature extraction__, while the upper levels focus on __consolidating these features to the output__.

## Usage
```python
model_A = keras.models.load_model("my_model_A.h5")

# We use the slice operator to get every layer but the last
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid")) # add Output Layer
```

<font color = red>__NOTE :__</font> `model_B_on_A` shares layers with `model_A`, since we did not deep copy the layers. If we deep copy the model using `clone_model()`, we can circumvent this problem as:

```python
model_A_clone = keras.models.clone_model(model_A) # Copies the structure
model_A_clone.set_weights(model_A.get_weights())  # Copies the weights
```

### Caveat
The transferred model will produce errors during the first few epochs atleast, because the output layer is randomly initialized. It does not help to change the resued layers however, since they are already mostly tuned to the task. So, we can "freeze" those layers so that the model trains better.

```python
for layer in model_B_on_A.layers[:-1]:     
	layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",                      metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B)
```

Once we have trained it for a few epochs, the model is in a better position to be trained as a whole. As a result it can now be trained together to get the final model.

```python
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-2
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))
```

<font color=red>__NOTE:__</font> Transfer learning generally does not work well for small dense networks, because they are small and learn patterns very specific to the task. They work well with deep [[Convolutional Neural Networks]], where patterns learned are more generally.