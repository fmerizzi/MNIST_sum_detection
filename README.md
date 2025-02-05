# MNIST Sum Detection with Clutter
This simple generated dataset is built on MNIST and FASHION-MNIST, it consists in an image of size 128x128 which contains < 5 MNIST digits and < 10 FASHION-MNIST clutter. 
The neural task consist in predicting the sum of the MNIST digit's value present the image. 

![example](https://github.com/fmerizzi/MNIST_sum_detection/blob/main/example_element.png)

For example, in this image the result is 24. 

## Generator

The dataset is comprised of a generator class which produces the data on the fly, it can be directly inserted in keras fit methods. 

## Custom Metric: Sum Accuracy
The evaluation metric computes the percentage of test samples where the integer-rounded model output matches the ground truth sum.

```python
def sum_metric(y_true, y_pred):
    # Ensure y_true is a tensor and has the same shape as y_pred
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.reshape(y_pred, tf.shape(y_true))  # Reshape to match dimensions
    
    # Round y_pred to nearest integer
    y_pred_rounded = tf.round(y_pred)

    # Compare elements
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32))

    # Compute final metric (accuracy-like)
    return correct_predictions / tf.cast(tf.size(y_true), tf.float32)

```

## Baseline model
A simple baseline model is given, a convolutional decoder. 

```python
# CNN Baseline Model
def build_baseline_model():
    inputs = keras.Input(shape=(128, 128, 1))
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='relu')(x)
    return keras.Model(inputs, outputs)
```
The model achieves a score of 30% on the sum accuracy metric. 

![example](https://github.com/fmerizzi/MNIST_sum_detection/blob/main/example_run.png)

