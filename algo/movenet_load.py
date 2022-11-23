# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

# Load the input image.
def load_image(path : str):
    """
    Take the path (as a string)of an image load and prepare it to be ingested by the model
    MoveNet Multipose Lightning 1
    input : path as a string
    output : tensorflow tensor 256 by 256 RGB with tf.int32 values
    """
    image = tf.io.read_file(path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 256), dtype=tf.int32)
    return image

# Download the model from TF Hub.
def load_model(mode:str ='local'):
    """
    load model from tensorflow hub and make it ready for porediction
    input : 'hub' or 'local'
    output : tensorflow model """
    if mode == 'hub':
        model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
        movenet = model.signatures['serving_default']
        return movenet
    else:
        model = tf.saved_model.load("../model/saved_model.pb")
        return model
