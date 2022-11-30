from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import subprocess
import os
import uuid
import cv2
import numpy as np

from api.script.movenet_load import load_video_and_release, load_model, predict_on_stream


app = FastAPI()
app.state.model = load_model(mode='hub')


# set up google cloud
SYNC_BUCKET = 'sync_testinput'
gcs = storage.Client()
bucket = gcs.get_bucket(SYNC_BUCKET)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return {'greeting': 'Hello from in_sync'}


@app.post("/vid_stats")
def stats_to_st(file: UploadFile = File(...)):
    # video file loading
    vid_name = file.filename
    uploaded_video = file.file
    output_name = 'output'


    # open video file
    with open(vid_name, mode='wb') as f:
        f.write(uploaded_video.read())


    cap = cv2.VideoCapture(vid_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    return {
        'frame_count': frame_count,
        'fps': fps,
        'dim': f'{width} x {height}',
        'width': width,
        'height': height,
        'vid_name': vid_name,
        'output_name': output_name
            }


@app.get("/vid_process")
def process_vid(vid_name, output_name, frame_count, fps, width, height, dancers):
    vid, writer, _, _, _, _ = load_video_and_release(vid_name, output_format="mp4", output_name=output_name)

    #return vid , all_scores, all_people, all_link_mae , worst_link_scores , worst_link_names
    vid, all_scores, _, _, worst_link_scores, worst_link_names = predict_on_stream(vid, writer, app.state.model, int(width), int(height), int(dancers))
    timestamps = np.arange(int(frame_count))/int(fps) #time in seconds

    # compress video output to smaller size
    my_uuid = uuid.uuid4()
    output_lite = f'output_lite_{my_uuid}.mp4'
    current_dir = os.path.abspath('.')
    result = subprocess.run(f'ffmpeg -i {current_dir}/{output_name}.mp4 -b:v 2500k {current_dir}/{output_lite} -y', shell=True)
    print(result)


    # upload video to google cloud storage
    vid_blob = bucket.blob(output_lite)
    vid_blob.upload_from_filename(output_lite)

    # # clean screencaps in google cloud storage
    # blobs = bucket.list_blobs(prefix='screencaps')
    # for blob in blobs:
    #     blob.delete()

    # upload screencaps to google cloud storage
    for i in range(int(frame_count)):
        blob = bucket.blob(f"screencaps/{my_uuid}/frame{i}.jpg")
        blob.upload_from_filename(f"{os.path.abspath('.')}/api/screencaps/frame{i}.jpg")


    return {
        'output_url': vid_blob.public_url,
        'timestamps': list(timestamps),
        'scores': all_scores,
        'my_uuid': my_uuid
        'link_scores': worst_link_scores,
        'link_names': worst_link_names
    }
