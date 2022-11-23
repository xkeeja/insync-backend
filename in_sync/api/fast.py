from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import cv2 as cv

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


@app.get("/vid_process")
def process(frame_count: int):
    frames = []
    
    for i in range(1, frame_count+1):
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'{i}.jpg')
        blob.download_to_filename(f'{i}.jpg')
        
        oriImg = cv.imread(f'{i}.jpg')
        frames.append(oriImg)
        
    return {'frame_count': len(frames)}