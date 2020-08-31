# Prefetching with Data API
Prefetching is another feature of the [[Data API]], where it tries to stay one batch ahead of the current training loop.

In other words, as one Dataset is training, another is being fetched to be used for the next epoch.

It can be evoked with the `tf.data.Dataset.prefetch()` function as:

```python
batch = dataset.batch(batch_size).prefetch(1)
```