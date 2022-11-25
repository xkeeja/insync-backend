from fastapi import FastAPI, UploadFile, File
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
SYNC_BUCKET = 'sync_testinput'


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
        'vid_name': vid_name,
        'output_name': output_name
            }
    

@app.get("/vid_processed")
def process_vid(vid_name, output_name, frame_count, fps):    
    vid, writer, _, _, _, _ = load_video_and_release(vid_name, output_format="mp4", output_name=output_name)
    
    vid, all_scores = predict_on_stream(vid, writer, app.state.model)
    timestamps = np.arange(int(frame_count))/int(fps) #time in seconds
    
    # compress video output to smaller size
    my_uuid = uuid.uuid4()
    output_lite = f'output_lite_{my_uuid}.mp4'
    current_dir = os.path.abspath('.')
    result = subprocess.run(f'ffmpeg -i {current_dir}/{output_name}.mp4 -b 800k {current_dir}/{output_lite} -y', shell=True)
    print(result)
    
    
    # upload video to google cloud storage
    gcs = storage.Client()
    bucket = gcs.get_bucket(SYNC_BUCKET)
    blob = bucket.blob(output_lite)
    blob.upload_from_filename(output_lite)
    
    
    return {
        'output_url': blob.public_url,
        'timestamps': list(timestamps),
        'scores': all_scores
    }