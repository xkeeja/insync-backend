import streamlit as st
import requests

#Receive video file from user upload
uploaded_video = st.file_uploader("**Upload video for evaluation**", type=(['gif']))

#If video has been uploaded
if uploaded_video:
    url = "http://127.0.0.1:8000/vid_process_from_st"
    files = {"file": (uploaded_video.name, uploaded_video, "multipart/form-data")}
    response = requests.post(url, files=files).json()
    
    st.write(response['frame_count'])