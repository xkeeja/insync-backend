# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from colorama import Fore, Style
import time

#Import calculation functions
from algo.calculations import data_to_people, similarity_scorer


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
    image = tf.cast(tf.image.resize_with_pad(image, 160, 256), dtype=tf.int32)
    return image

# Download the model from TF Hub.
def load_model(mode:str ='local'):
    """
    load model from tensorflow hub and make it ready for porediction
    input : 'hub' or 'local'
    output : tensorflow model """

    start=time.time()
    if mode == 'hub':
        model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
        model = model.signatures['serving_default']
    else:
        model = tf.saved_model.load("../model/saved_model.pb")
    print(Fore.BLUE + f"model loads in: {time.time()-start}s" + Style.RESET_ALL)
    return model


def load_video_and_release(path : str, output_format: str, output_name :str):
    """

    """
    # Conversion on the video in a opencv Videocapture (collection of frames)
    vid = cv2.VideoCapture(path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video analysed: \n fps: {fps}, *\
          \n frame count: {frame_count} , \n width : {width}, \n height : {height}")

    # creation onf the writer to recompose the video later on
    if output_format =="avi":
        writer = cv2.VideoWriter(f"{output_name}.avi",
        cv2.VideoWriter_fourcc(*"MJPG"), fps,(width,height))
    elif output_format =="mp4":
        writer = cv2.VideoWriter(f"{output_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps,(width,height))

    return vid, writer, fps, frame_count, width, height

def preprocess_image(image, new_width, new_height):
    """
    take an frame of a video converted to an image through opencv,
    wth the new_width and new height  for reshaping purpose.
    Based on the image original definition :
    - (480p: 854px by 480px)
    - (720p: 854px by 480px)
    - (1080p: 854px by 480px)
    """
    resize_table = {480}
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
    """
    Use the model to predict the keypoints given a reshaped input_image.
    """
    # Run model inference.
    start = time.time()
    outputs = model(input_image)
    # Output is a [1, 6, 56] tensor that we can reshape
    keypoints = outputs['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    print(Fore.BLUE + f"Prediction and keypoint output in: {time.time()-start}s" + Style.RESET_ALL)
    return keypoints

def drawing_joints(keypoints, number_people , frame):
    """
    Plot the positions of the joints on a frame.
    """
    start=time.time()
    for person_id in range(number_people):
        print(np.mean(keypoints[person_id,:,2]))
        if np.mean(keypoints[person_id,:,2]) < 0.1:
            pass
        else:
            print("plotting ", person_id)
            x_vals = keypoints[person_id,:,1]
            y_vals = keypoints[person_id,:,0]
            #print (x_vals, type(x_vals))
            for x,y in zip(x_vals, y_vals):
                cv2.circle(
                img=frame,
                center=(int(x), int(y)),
                radius=7,
                color=(255,255,255),
                thickness=-1,
                lineType=cv2.LINE_AA
                )
                cv2.circle(
                img=frame,
                center=(int(x), int(y)),
                radius=5,
                color=(120,10,120),
                thickness=-1,
                lineType=cv2.LINE_AA
                )
                # frame = cv2.drawMarker(
                #     img=frame,
                #     position = (int(x),int(y)),
                #     color=(255,255,255),
                #     markerType=cv2.MARKER_CROSS,
                #     markerSize= 23,
                #     thickness= 3,
                #     line_type=8
                # )
                # frame = cv2.drawMarker(
                #     img=frame,
                #     position = (int(x),int(y)),
                #     color=(120,10,120),
                #     markerType=cv2.MARKER_CROSS,
                #     markerSize= 20,
                #     thickness= 3,
                #     line_type=8
                # )
    print(Fore.BLUE + f"Plotting joints output made in: {time.time()-start}s" + Style.RESET_ALL)
    return frame

def drawing_links(people, link_mae, frame, linkwidth: int):
    """
    Plot the line of the links based on a treshold value for color
    """
    start=time.time()
    for person in people:
        for link in person.links:
            mae = link_mae[link.id]
            if mae>=30:
                link.set_color((28,25,215))# red in BGR channel (opencv swap the channels)
            elif mae>=20:
                link.set_color((97,174,253))
            elif mae>=10:
                link.set_color((191,255,255))
            elif mae>=5:
                link.set_color((106,217,166))

            else:
                link.set_color((65,150,26))

            x1 , y1 = int(link.joints[0].x), int(link.joints[0].y)
            x2 , y2 = int(link.joints[1].x), int(link.joints[1].y)
            X_mean = int((x1+x2)/2)
            Y_mean = int((y1+y2)/2)
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = link.angle
            polygon = cv2.ellipse2Poly(
                center=(X_mean,Y_mean),
                axes=(int(length/2),linkwidth),
                angle= int(angle),
                arcStart=0,
                arcEnd=360,
                delta=1
            )
            cv2.fillConvexPoly(
                img=frame,
                points=polygon,
                color=link.color,
                lineType=cv2.LINE_AA
            )

            # frame = cv2.line(img=frame,
            #                 pt1=(x1, y1),
            #                 pt2=(x2, y2),
            #                 color=link.color,
            #                 thickness=5,
            #                 lineType=cv2.LINE_AA
            # )
    print(Fore.BLUE + f"Plotting link output made in: {time.time()-start}s" + Style.RESET_ALL)
    return frame


def add_frame_text(frame, count: int):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img=frame,
                       text=f'{count}',
                       org=(10,50),
                       fontFace=font,
                       fontScale=2,
                       color=(0, 255, 0),
                       thickness=2,
                       lineType=cv2.LINE_AA,
                       bottomLeftOrigin=False)


def calculate_score(keypoints , number_of_people):
    """
    Calculate the angles between joints given the keypoints.
    Give a similariy score for the the frame.
    """
    start = time.time()
    people =  data_to_people(keypoints , number_of_people)
    link_mae, frame_score = similarity_scorer(people)
    print(Fore.BLUE + f"Scoring completed in: {time.time()-start}s" + Style.RESET_ALL)
    return people, link_mae, frame_score


def predict_on_stream (vid, writer, model, width, height):
    """

    """
    all_scores = []
    all_people = []
    all_link_mae = []
    count = 0
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret==True:
            count+=1
            image = frame.copy()
            #Preprocessing the image
            input_image = preprocess_image(image, 256, 256)
            # making prediction
            keypoints = predict(model, input_image)
            keypoints= np.squeeze(np.multiply(keypoints, [height,width,1]))
            #print(keypoints)
            #Calculate scores

            people, link_mae, frame_score  = calculate_score(keypoints , number_of_people=2)
            all_scores.append(frame_score)
            all_people.append(people)
            all_link_mae.append(link_mae)
            max_id = np.argmax(link_mae)
            name_link_max = people[0].links[max_id].name

            print(f"FRAME_SCORE{frame_score}, MAX_LINK:({max_id} : {name_link_max})")
            #frame = cv2.flip(frame,0)
            image = drawing_links(people, link_mae, image, linkwidth=6)
            frame_mask = image.copy()
            frame_mask = drawing_joints(keypoints, number_people=2, frame=frame_mask)

            frame_superposition = cv2.addWeighted(src1=frame,
                                                  alpha=0.35,
                                                  src2=frame_mask,
                                                  beta=0.65,
                                                  gamma=0)

            frame_resize = cv2.resize(
                    frame_superposition,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4
            ) # OpenCV processes BGR images instead of RGB
            frame_text = add_frame_text(frame_resize, count)

            writer.write(frame_text)
        else:
            break

    writer.release()

    return vid , all_scores, all_people, all_link_mae
