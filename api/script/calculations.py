import time
import numpy as np

from api.script.person import Joint


def calculate_angle( joint1, joint2):
    """takes two joint objects and returns the angle between them
    angle of 2 seen from 1 """
    x1, y1 = joint1.x  , joint1.y
    x2, y2 = joint2.x , joint2.y
    #confidence1, confidence2= joint1_id[2], joint2_id[2]

    delta_x = x2-x1
    delta_y = y2-y1

    if delta_x ==0 and delta_y ==0:
        return None
    elif delta_x ==0:
        if delta_y >0:
            return 90
        else:
            return 270
    elif delta_y ==0:
        if delta_x>0 :
            return 0
        else:
            return 180
    elif delta_x >0 and delta_y>0:
        grad= abs(delta_y/delta_x)
        return np.arctan(grad)*180/np.pi
    elif delta_x  <0 and delta_y>0:
        grad = abs(delta_x/delta_y)
        return np.arctan(grad)*180/np.pi +90
    elif delta_x< 0 and delta_y< 0:
        grad = abs(delta_y/delta_x)
        return np.arctan(grad)*180/np.pi +180
    else:
        grad= abs(delta_x / delta_y)
        return np.arctan(grad)*180/np.pi +270


def return_angles(keypoints, number_of_people):
    """return angles of all links of all people"""

    connecting_body_parts_id=[(4,2),(2,0),(0,1),(1,3),(5,6),(5,7),(7,9),
                              (6,8),(8,10),(6,12),(5,11),(12,11),(12,14),
                              (11,13),(13,15),(14,16)]
    all_angles= []
    all_angles_with_joints = []

    for person in range(len(keypoints[:number_of_people])):
        person_angles= []
        links = []
        angles_with_joints = []
        for connection in connecting_body_parts_id:
            joint1 = Joint(connection[0] , person )
            joint2 = Joint(connection[1] , person)
            joint1.add_coord(keypoints[person, joint1.id,1] ,keypoints[person,joint1.id,0])
            joint2.add_coord(keypoints[person,joint2.id,1] ,keypoints[person,joint2.id,0])

            angle =calculate_angle( joint1, joint2)
            angle_with_joints= (angle, joint1, joint2)
            person_angles.append(angle)
            angles_with_joints.append(angle_with_joints)
        all_angles.append(person_angles)
        all_angles_with_joints.append(angles_with_joints)
    return all_angles , all_angles_with_joints


def similarity_scorer(angles , number_of_people , strictness=1 ):
    """returns difference if two people and variance if more than two people"""
    link_scores_dict ={}
    link_scores_list =[]
    if number_of_people ==2:
        for link_angle in range(16):
            if link_angle <4:
                continue

            link_score = abs(angles[1][link_angle]-angles[0][link_angle])**strictness
            link_scores_dict[link_angle]= link_score
            link_scores_list.append(link_score)

    else:
        for link_angle in range(16):
            mu = np.mean(angles[:,link_angle])
            link_score = (abs(angles[:,link_angle]- mu))**strictness
            link_scores_dict[link_angle] = link_score
            link_scores_list.append(link_score)


    frame_score = (np.sum(link_scores_list)**(1/strictness))/17
    max = 0
    max_link = 0
    for key, val in link_scores_dict.items():
        if val > max:
            max =val
            max_link = key

    return link_scores_list , frame_score , max_link , max, link_scores_dict
