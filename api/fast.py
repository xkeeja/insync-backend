from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv


app = FastAPI()
# app.state.model = load_model()
# BUCKET_NAME = 'sync_testinput'


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
    vidcap = cv.VideoCapture(vid_name)
    frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    
    return {'frame_count': frame_count} 