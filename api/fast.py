from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import cv2 as cv
from PIL import Image
import numpy as np

app = FastAPI()
# app.state.model = load_model()

BUCKET_NAME = 'sync_testinput'

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


# @app.get("/vid_process_from_bucket")
# def process_from_bucket(frame_count: int):
#     frames = []
    
#     for i in range(1, frame_count+1):
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blob = bucket.blob(f'{i}.jpg')
#         blob.download_to_filename(f'{i}.jpg')
        
#         oriImg = cv.imread(f'{i}.jpg')
#         frames.append(oriImg)
        
#     return {'frame_count': len(frames)}

@app.post("/vid_process_from_st")
def process_from_st(file: UploadFile = File(...)):
    vid_name = file.filename
    uploaded_video = file.file
    with open(vid_name, mode='wb') as f:
        f.write(uploaded_video.read())
        
    vidcap = cv.VideoCapture(vid_name)
    frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    
    return {'frame_count': frame_count} 