import streamlit as st
import requests
import time

#Receive video file from user upload
uploaded_video = st.file_uploader("**Upload video for evaluation**", type=(['mp4']))

#If video has been uploaded
if uploaded_video:
    url = "http://127.0.0.1:8000/vid_process_from_st"
    # url = "https://sync-eagwezifvq-an.a.run.app/vid_process_from_st"
    files = {"file": (uploaded_video.name, uploaded_video, "multipart/form-data")}
    
    start_time = time.time()
    response = requests.post(url, files=files).json()
    
    st.write(response)
    st.write("--- %s seconds ---" % (time.time() - start_time))