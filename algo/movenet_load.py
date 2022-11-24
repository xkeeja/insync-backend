# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy
from colorama import Fore, Style
import time

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
    image = tf.cast(tf.image.resize_with_pad(image, 256, 448), dtype=tf.int32)
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

def preprocess_image(image, new_width, new_height):
    start = time.time()
    image = cv2.resize(image, (new_width, new_height))
    # Resize to the target shape and cast to an int32 vector
    input_image = tf.cast(tf.image.resize_with_pad(image, new_width, new_height), dtype=tf.int32)
    # Create a batch (input tensor)
    input_image = tf.expand_dims(input_image, axis=0)

    print(Fore.BLUE + f"image processed in: {time.time()-start}s" + Style.RESET_ALL)
    print(input_image.shape)
    return input_image

def predict(model, input_image):
    # Run model inference.
    start = time.time()
    outputs = model(input_image)
    # Output is a [1, 6, 56] tensor that we can reshape
    keypoints = outputs['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    print(Fore.BLUE + f"Prediction and keypoint output in: {time.time()-start}s" + Style.RESET_ALL)
    return keypoints

def load_video_and_release(path : str, output_format: str, output_name :str):

    # COnversion on the video in a opencv Videocapture (collection of frames)
    vid = cv2.VideoCapture(path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video analysed: /n fps: {fps}, *\
          /n frame count: {frame_count} , /n width : {width}, height : {height}")

    # creation onf the writer to recompose the video later on
    if output_format =="avi":
        writer = cv2.VideoWriter(f"{output_name}.avi",
        cv2.VideoWriter_fourcc(*"MJPG"), fps,(width,height))
    elif output_format =="mp4":
        writer = cv2.VideoWriter(f"{output_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps,(width,height))
    return vid, writer, fps, frame_count, width, height

def predict_on_stream (vid, writer, model):
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret==True:
            image = frame.copy()
            input_image = preprocess_image(image, 256, 448)
            predict(model, input_image)
            frame = cv2.flip(frame,0)
            writer.write(frame)
        else:
            break
    writer.release()
    return vid
