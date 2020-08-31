# Performance Scheduling

Measure the validation error every $N$ steps (just like [[Early Stopping]]) and then reduce the learning rate by a factor of $\lambda$ when error stops dropping.

## Usage 

Keras offers the `ReduceLROnPlateau` callback.

```python
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience=5)
```

Now, `lr_scheduler` can be used as a callback.