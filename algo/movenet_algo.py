import time
import numpy as np
from movenet_data import run_inference
from matplotlib import pyplot as plt
import matplotlib.image as mpimg



def calculate_angles(feature1, feature2):
    x1 , y1 = feature1[0] , feature1[1]
    x2 , y2 = feature2[0], feature2[1]
    score1, score2= feature1[2], feature2[2]

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


def return_angles(data, number_of_people):
    """Return array of lists with gradients of connecting body parts
    LEFT = LEFT FROM VIEWER
    connecting_body_parts= {"left eye to left ear":(2,4),
                             "left eye to nose":(2,0),
                             "nose to right eye":(0,1),
                             "right eye to right ear":(1,3),
                             "right shoulder to left shoulder":(5,6),
                             "left shoulder to left elbow":(8,6),
                             "left elbow to left wrist" :(8,10),
                             "right shoulder to right elbow":(5,7),
                             "right elbow to right wrist":(7,9),
                             "left shoulder to left hip":(6,12),
                             "chest to right hip":(5,11),
                             "left hip to right hip":(11,12),
                             "left hip to left knee":(12,14),
                             "left knee to left foot":(14,16),
                             "right hip to right knee":(11,13),
                             "right knee to right foot":(13,15)}
    """


    connecting_body_parts_id=[(4,2),(2,0),(0,1),(1,3),(5,6),(5,7),(7,9),
                              (6,8),(8,10),(6,12),(5,11),(12,11),(12,14),
                              (11,13),(13,15),(14,16)]
    output = []

    for person in data[:number_of_people]:
        person_angles= []
        for connection in connecting_body_parts_id:
            feature_1 = person[connection[0]]
            feature_2 = person[connection[1]]
            angle =calculate_angles(feature_1, feature_2)

            person_angles.append(angle)

        output.append(person_angles)

    return output



start=time.time()
output_frames, keypoints= run_inference()
end = time.time()
print("model time ", end-start,
      "per frame" ,(end-start)/95)


start = time.time()
angles = np.array(return_angles(keypoints,2))
end = time.time()
print("time:" ,end-start)
print(angles.shape)
for x in range(16):
    diff = angles[0,x]- angles[1,x]
    print(diff)


import seaborn as sns
import matplotlib.image as mpimg
img = mpimg.imread("raw_data/6people.webp")
plt.imshow(img)

features = range(17)


print(keypoints.shape)



print(keypoints)

x_vals_1 = keypoints[0,:,1]*390
y_vals_1 = keypoints[0,:,0]*280

x_vals_2 = keypoints[1,:,1]*390
y_vals_2 = keypoints[1,:,0]*280

x_vals_3 = keypoints[2,:,1]*390
y_vals_3 = keypoints[2,:,0]*280

x_vals_4 = keypoints[3,:,1]*390
y_vals_4 = keypoints[3,:,0]*280

x_vals_5 = keypoints[4,:,1]*390
y_vals_5 = keypoints[4,:,0]*280

x_vals_6 = keypoints[5,:,1]*390
y_vals_6 = keypoints[5,:,0]*280


sns.scatterplot(x=x_vals_1, y=y_vals_1, hue =features)

sns.scatterplot(x=x_vals_2, y=y_vals_2, hue=features)
sns.scatterplot(x=x_vals_3, y=y_vals_3, hue =features)

sns.scatterplot(x=x_vals_4, y=y_vals_4, hue=features)
sns.scatterplot(x=x_vals_5, y=y_vals_5, hue =features)

sns.scatterplot(x=x_vals_6, y=y_vals_6, hue=features)

plt.show()
