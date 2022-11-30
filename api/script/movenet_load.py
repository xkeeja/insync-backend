# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from colorama import Fore, Style
import time
import os
import glob

#Import calculation functions
from api.script.calculations import data_to_people, similarity_scorer


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

def drawing_joints(keypoints, people , frame, confidence_display=True):
    """
    Plot the positions of the joints on a frame.
    """
    number_people = len(people) # number of people selected by the user
    no_display = people[0].joints_to_not_be_displayed()
    start=time.time()
    for person_id in range(number_people):
        print(np.mean(keypoints[person_id,:,2]))
        if np.mean(keypoints[person_id,:,2]) < 0.1:
            pass
        else:
            for person in people:
                print("plotting ", person.id)
                for joint, display_off in zip(person.joints, no_display):
                    if display_off:
                        pass
                    else:
                        x = joint.x
                        y = joint.y
                        conf = round(joint.confidence,4)
                        cv2.circle(
                        img=frame,
                        center=(int(x), int(y)),
                        radius=14,
                        color=(255,255,255),
                        thickness=-1,
                        lineType=cv2.LINE_AA
                        )
                        cv2.circle(
                        img=frame,
                        center=(int(x), int(y)),
                        radius=12,
                        color=(120,10,120),
                        thickness=-1,
                        lineType=cv2.LINE_AA
                        )
                        if confidence_display:
                            X_top_box = int(x)-7
                            Y_top_box = int(y)-15
                            X_bottom_box = int(x)+65
                            Y_bottom_box = int(y)+4


                            #background rectangle for the confidence score display per joint
                            cv2.rectangle(
                                img=frame,
                                pt1=(X_top_box,Y_top_box), # top left corner
                                pt2=(X_bottom_box,Y_bottom_box),#bottom right corner
                                color=(255,255,255),
                                thickness=-1,
                                lineType=cv2.LINE_AA
                            )
                            #confidence score display per joint
                            cv2.putText(
                                img=frame,
                                text=f'{conf}',
                                org=(int(x)-5,int(y)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0, 0, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA,
                                bottomLeftOrigin=False
                            )
                #########################################################

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
                    ###############################################
    print(Fore.BLUE + f"Plotting joints output made in: {time.time()-start}s" + Style.RESET_ALL)
    return frame

def drawing_links(people, link_mae, frame, linkwidth: int):
    """
    Plot the line of the links based on a treshold value for color
    """
    face_ignored = people[0].face_ignored
    start=time.time()
    for person in people:
        for i, link in enumerate(person.links):
            mae = link_mae[i]
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


def add_frame_text(frame, count: int, color:tuple):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img=frame,
                       text=f'{count}',
                       org=(10,50),
                       fontFace=font,
                       fontScale=2,
                       color=color,
                       thickness=2,
                       lineType=cv2.LINE_AA,
                       bottomLeftOrigin=False)


def calculate_score(keypoints , number_of_people:int, face_ignored:bool, conf_threshold:float):
    """
    Calculate the angles between joints given the keypoints.
    Give a similariy score for the the frame.
    """
    start = time.time()
    people =  data_to_people(keypoints , number_of_people, face_ignored)
    link_mae, frame_score, worst_link_name, worst_link_score, ignore_for_display = similarity_scorer(people, conf_threshold)
    print(Fore.BLUE + f"Scoring completed in: {time.time()-start}s" + Style.RESET_ALL)
    return people, link_mae, frame_score , worst_link_name , worst_link_score, ignore_for_display


def predict_on_stream (vid, writer, model, width: int, height :int,
                       dancers:int, face_ignored:bool, conf_threshold:float):
    """

    """
    all_scores = []
    all_people = []
    all_link_mae = []
    worst_link_scores =[]
    worst_link_names =[]
    frame_is_ignored_list=[]
    count = 0

    # clean screencaps folder
    files = glob.glob(f"{os.path.abspath('.')}/api/screencaps/*.jpg")
    for f in files:
        os.remove(f)

    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret==True:
            image = frame.copy()
            #Preprocessing the image
            input_image = preprocess_image(image, 256, 256)
            # making prediction
            keypoints = predict(model, input_image)
            keypoints= np.squeeze(np.multiply(keypoints, [height,width,1]))
            #print(keypoints)
            #Calculate scores

            people, link_mae, frame_score, worst_link_name, \
                worst_link_score, ignore_for_display = calculate_score(
                    keypoints=keypoints,
                    number_of_people=dancers,
                    face_ignored=face_ignored,
                    conf_threshold=conf_threshold
                    )
            all_scores.append(frame_score)
            all_people.append(people)
            all_link_mae.append(link_mae)
            worst_link_scores.append(worst_link_score)
            worst_link_names.append(worst_link_name)
            frame_is_ignored_list.append(ignore_for_display)

            if ignore_for_display:
                frame_resize = cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4
                )
                frame_text = add_frame_text(frame_resize, "not analysed", color=(0, 0, 255))

            else:

                print(f"FRAME_SCORE{frame_score}, WORST LINK_NAME:{worst_link_name}, WORST LINK SCORE: {worst_link_score}")
                #frame = cv2.flip(frame,0)
                image = drawing_links(people, link_mae, image, linkwidth=6)
                frame_mask = image.copy()
                people = all_people[count]
                frame_mask = drawing_joints(keypoints, people=people, frame=frame_mask)

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
                frame_text = add_frame_text(frame_resize, count, color=(0, 255, 0))

            cv2.imwrite(f"{os.path.abspath('.')}/api/screencaps/frame%d.jpg" % count, frame_text)
            count += 1

            writer.write(frame_text)
        else:
            break

    writer.release()

    return vid , all_scores, all_people, all_link_mae , worst_link_scores , worst_link_names
