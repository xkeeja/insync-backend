from posevisual.person import Joint
from movenet_load import load_image, load_model
from matplotlib import pyplot as plt
import numpy as np
# import seaborn as sns
import matplotlib.image as mpimg
import os


def calculate_angles(keypoints , joint1_id, joint2_id):
    x1 , y1 = keypoints[joint1_id][0] ,keypoints[joint1_id][1]
    x2 , y2 = keypoints[joint2_id][0] ,keypoints[joint2_id][1]
    confidence1, confidence2= joint1_id[2], joint2_id[2]
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
            angle =calculate_angles(keypoints , joint1.id, joint2.id)
            if joint1 not in joints:
                joints.append(joint1)
            if joint2 not in joints:
                joints.append(joint2)
            person_angles.append(angle)
        all_angles.append(person_angles)
    return all_angles , joints, links

image = mpimg.imread("./algo/test_image/6people_2.jpg")
plt.imshow(image)
plt.show()

"""
path = os.path.join("..")
image_processed = load_image("/test_image/6people_2.jpg")
movenet = load_model()

# Run model inference.
outputs = movenet(image)
# Output is a [1, 6, 56] tensor.
keypoints = outputs['output_0']

print (keypoints)
"""
# all_angles, joints, links = return_angles(keypoints,6)
# x_vals_1 = all_angles[0,:,1]*390
# y_vals_1 = all_angles[0,:,0]*480
# x_vals_2 = all_angles[1,:,1]*390
# y_vals_2 = all_angles[1,:,0]*480
# x_vals_3 = all_angles[2,:,1]*390
# y_vals_3 = all_angles[2,:,0]*280
# x_vals_4 = all_angles[3,:,1]*390
# y_vals_4 = all_angles[3,:,0]*280
# x_vals_5 = all_angles[4,:,1]*390
# y_vals_5 = all_angles[4,:,0]*280
# x_vals_6 = all_angles[5,:,1]*390
# y_vals_6 = all_angles[5,:,0]*280

# img = mpimg.imread("test_image/6people.webp")
# plt.imshow(img)
# fig = plt.gcf()
# #fig.scatter(x=x_vals_1, y=y_vals_1, hue =joints.id)
# # sns.scatterplot(x=x_vals_2, y=y_vals_2, hue=joints.id)
# # sns.scatterplot(x=x_vals_3, y=y_vals_3, hue =joints.id)
# # sns.scatterplot(x=x_vals_4, y=y_vals_4, hue=joints.id)
# # sns.scatterplot(x=x_vals_5, y=y_vals_5, hue =joints.id)
# # sns.scatterplot(x=x_vals_6, y=y_vals_6, hue=joints.id)
# plt.show()


# # Run model inference.
# outputs = movenet(image)
# # Output is a [1, 6, 56] tensor.
# keypoints = outputs['output_0']
