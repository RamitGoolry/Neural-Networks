# Preprocessing with the Data API

## Chaining
Chaining is done with the `repeat()` function. This is similar to list multiplication in vanilla multiplication.

We can then use the `batch()` function to split it into multiple batches.

```python
dataset = tf.data.Dataset.range(10).repeat(3)
```

## Shuffling Data
[[Gradient Descent]] works better when data is Independent and Identically Distributed. Therefore, it is a good idea to use the `shuffle()` method. It will create a new shuffled dataset.

```python
dataset = dataset.shuffle(buffer_size = 5, seed = 42).batch(7)
```

For a large dataset which does not fit into memory, a simple shuffle does not work, since the buffer may be too small compared to the dataset as a whole. In this instance, it is better to shuffle the source dataset. This can be done on Mac using the [gshuf command](https://superuser.com/questions/760732/randomly-shuffle-rows-in-a-large-text-file). 

## Interleaving

It also makes sense to split the source data into multiple files, choose a few files randomly and interleave inputs from them, just in case a single (or multiple) shuffles didn't work well. 

You can do this with the `interleave()` function. 

```python
filepath_dataset = tf.data.Dataset.list_files(train_filepaths)

dataset = filepath_dataset.interleave(
				lambda filepath : tf.data.TextLineDatasets(filepath).skip(1) # Skips header,
				cycle_length = 5
			)
```

By default, interliave does not use parallelism, but that can be invoked with the `num_parallel_calls` parameter.

## Map
Many functions can be applied to the dataset through the `map()` function, which acts as the map HOF from vanilla python.