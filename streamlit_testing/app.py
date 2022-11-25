import streamlit as st
import requests
import time

#Receive video file from user upload
uploaded_video = st.file_uploader("**Upload video for evaluation**", type=(['mp4']))

#If video has been uploaded
if uploaded_video:
    url = "http://127.0.0.1:8000/vid_stats"
    files = {"file": (uploaded_video.name, uploaded_video, "multipart/form-data")}
    
    start_time = time.time()
    post_response = requests.post(url, files=files).json()
    
    st.write(post_response)
    st.write("--- %s seconds ---" % (time.time() - start_time))
    
    st.video(uploaded_video)
    

    if post_response:
        if st.button('Analyse synchronization'):
            
            st.write('Processing...')
            url = "http://127.0.0.1:8000/vid_processed"
            params = {"vid_name": post_response['vid_name'], "output_name": post_response['output_name']}
            
            start_time = time.time()
            get_response = requests.get(url, params=params).json()
            
            st.write(get_response)
            st.write("--- %s seconds ---" % (time.time() - start_time))


            st.video(get_response['output_url'])