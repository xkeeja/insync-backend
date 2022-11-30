import cv2
import numpy as np
from api.script.movenet_load import load_video_and_release, load_model, predict_on_stream

# cap = cv2.VideoCapture("Duet_clip_1_In_Sync.mp4")
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"fps: {fps}, frame count: {frame_count} , {width}, {height}")

# writer = cv2.VideoWriter("output.avi",
# cv2.VideoWriter_fourcc(*"MJPG"), fps,(width,height))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)
#         writer.write(frame)
#     else:
#         break

# # for frame in range(1000):
# #     writer.write(np.random.randint(0, 255, (480,640,3)).astype('uint8'))

# writer.release()

model = load_model(mode="hub")

vid, writer, fps, \
    frame_count, width, height = load_video_and_release("dancingvid.mp4",
                                                  output_format="mp4",
                                                  output_name="output_stream_3")
vid , all_scores, all_people, all_link_mae , worst_link_scores, \
    worst_link_names = predict_on_stream(vid, writer, model, width, height, 2,\
        face_ignored=True, conf_threshold=0.20)
timeline = np.arange(frame_count)/fps #time in seconds

print(all_scores)
print(all_link_mae)
print(worst_link_scores)
print(worst_link_names)
a = np.array(all_scores)
num_not_analysed_frames = len(a[a==None])
per = round(num_not_analysed_frames *100/ frame_count, 1)

print(f"out of {frame_count}, {num_not_analysed_frames} ({per}) were not analysed")
