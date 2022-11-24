from posevisual.person import Joint
from movenet_load import load_image, load_model
from matplotlib import pyplot as plt
import time
import numpy as np
# import seaborn as sns
import matplotlib.image as mpimg
from colorama import Fore, Style
from matplotlib import collections  as mc


def calculate_angles(keypoints , joint1_id, joint2_id):
    x1 , y1 = keypoints[joint1_id][0] ,keypoints[joint1_id][1]
    x2 , y2 = keypoints[joint2_id][0] ,keypoints[joint2_id][1]
    #confidence1, confidence2= joint1_id[2], joint2_id[2]
    delta_x = x2-x1
    delta_y = y2-y1
    if delta_x ==0:
        if delta_y >0:
            return 90
        else:
            return 270
    elif delta_x >0 and delta_y>0:
        grad= abs((y2-y1)/(x2-x1))
        return np.arctan(grad)*180/np.pi
    elif delta_x  <0 and delta_y>0:
        grad = abs((y2-y1)/(x1-x2))
        return np.arctan(grad)*180/np.pi +90
    elif delta_x< 0 and delta_y< 0:
        grad = abs((y1-y2)/(x2-x1))
        return np.arctan(grad)*180/np.pi +180
    else:
        grad= abs((y2-y1)/(x2-x1))
        return np.arctan(grad)*180/np.pi +270




def return_angles(keypoints, number_of_people):
    """Return array of lists with gradients of connecting body parts"""
    connecting_body_parts_id=[(4,2),(2,0),(0,1),(1,3),(5,6),(5,7),(7,9),
                              (6,8),(8,10),(6,12),(5,11),(12,11),(12,14),
                              (11,13),(13,15),(14,16)]
    all_angles = []
    joints= []
    for person in keypoints[:number_of_people]:
        person_angles= []
        joints = []
        links = []
        for connection in connecting_body_parts_id:
            joint1 = Joint(connection[0] , person )
            joint2 = Joint(connection[1] , person)
            angle =calculate_angles(person , joint1.id, joint2.id)
            if joint1 not in joints:
                joints.append(joint1)
            if joint2 not in joints:
                joints.append(joint2)
            person_angles.append(angle)
        all_angles.append(person_angles)
    return all_angles , joints, links

# Preprocessed the picture
start = time.time()
image_processed = load_image("./algo/test_image/test_crossign_arms.jpg")
print(Fore.BLUE + f"image processed in: {time.time()-start}s" + Style.RESET_ALL)
print(image_processed.shape)
# loading the model
start = time.time()
movenet = load_model("hub")
print(Fore.BLUE + f"model loaded in: {time.time()-start}s" + Style.RESET_ALL)

# Run model inference.
start = time.time()
outputs = movenet(image_processed)
# Output is a [1, 6, 56] tensor that we can reshape
keypoints = outputs['output_0'].numpy()[:,:,:51].reshape((6,17,3))
print(Fore.BLUE + f"Prediction and keypoint output in: {time.time()-start}s" + Style.RESET_ALL)

print (keypoints)
number_people = 6


start = time.time()
all_angles, joints, links = return_angles(keypoints,number_people)
print(Fore.BLUE + f"angle calculation output in: {time.time()-start}s" + Style.RESET_ALL)
print(all_angles)

width , height = 1280 , 720


img = mpimg.imread("./algo/test_image/test_crossign_arms.jpg")
plt.imshow(img)
fig = plt.gcf()

start = time.time()
for person_id in range(number_people):
    print(np.mean(keypoints[person_id,:,2]))
    if np.mean(keypoints[person_id,:,2]) < 0.1:
        pass
    else:
        print("plotting ", person_id)
        x_vals = keypoints[person_id,:,1]*width
        y_vals = keypoints[person_id,:,0]*height
        plt.scatter(x=x_vals, y=y_vals, marker="+", color=[Joint(i, person_id).color for i in range(17)])
print(Fore.BLUE + f"Plotting output made in: {time.time()-start}s" + Style.RESET_ALL)

# lines = [[(130, 120), (400, 300)]]
# c = np.array([(1, 0, 0, 1)])

# lc = mc.LineCollection(lines, colors=c, linewidths=2)
# ax = plt.gca()
# ax.add_collection(lc)
plt.savefig("saved_figure.png")
