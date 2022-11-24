from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import cv2
import subprocess
import os


app = FastAPI()
# app.state.model = load_model()
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


@app.post("/vid_process_from_st")
def process_from_st(file: UploadFile = File(...)):
    # video file loading
    vid_name = file.filename
    uploaded_video = file.file
    
    # open video file
    with open(vid_name, mode='wb') as f:
        f.write(uploaded_video.read())
        
    # load video file into OpenCV
    # vidcap = cv.VideoCapture(vid_name)
    # frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    
    cap = cv2.VideoCapture(vid_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_name = "output.mp4"
    output_width = 854
    output_height = int(height * (output_width / width))
    writer = cv2.VideoWriter(output_name,
    cv2.VideoWriter_fourcc(*"mp4v"), fps,(output_width,output_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)
            frame = cv2.resize(frame, (output_width, output_height), interpolation = cv2.INTER_AREA)
            writer.write(frame)
        else:
            break
    
    writer.release()
    
    
    # compress video output to smaller size
    output_lite = 'output_lite.mp4'
    current_dir = os.path.abspath('.')
    result = subprocess.run(f'ffmpeg -i {current_dir}/{output_name} -b 800k {current_dir}/{output_lite} -y', shell=True)
    print(result)
    
    
    # upload video to google cloud storage
    gcs = storage.Client()
    bucket = gcs.get_bucket(SYNC_BUCKET)
    blob = bucket.blob(output_lite)
    blob.upload_from_filename(output_lite)
    
    blob_status = False
    while blob_status == False:
        blob_status = blob.exists()
    
    return {
        'frame_count': frame_count,
        'fps': fps,
        'dim': f'{width} x {height}',
        'dir': current_dir,
        'output_url': blob.public_url
            } 