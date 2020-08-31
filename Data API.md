# The Data API

When using very large datasets (larger than RAM), we can use the Data API to help us load in data in smaller batches, or from various sources. It also helps [[Preprocessing with Data API|preprocess]] the data, so that it can be used better, and more efficiently.

All of the Data API revolves around the idea of a dataset (`tf.data.Dataset`).