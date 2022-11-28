import numpy as np
from posevisual.person import Joint, Link , Person , link_def

"""Module for algorithm calculations."""




def calculate_angle( joint1:Joint, joint2:Joint):
    """takes two joint objects and returns the angle of 2 seen from 1
    y in opposite direction to convention"""
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


def data_to_people(keypoints: list, number_of_people:int):
    """List of people objects with coordinates, confidence and angles assigned to joints and links.
    """
    #Create list of person objects
    people = []
    keypoints= np.array(keypoints)
    for person_id in range(number_of_people):
        #Instantiate person
        person = Person(person_id)
        #Assign all the coordinates and confidence to the person
        person.update_joints(keypoints[person_id,:,1], keypoints[person_id,:,0],keypoints[person_id,:,2])
        person.create_links()
        for link in person.links:
            #Calculate angle
            link.add_angle(calculate_angle(link.joints[0], link.joints[1]))


        people.append(person)

    return people





def similarity_scorer(people:list):
    """
    Takes list of person objects
    Returns list of mean absolute error between angles for each link
    Returns overall frame score
    """
    number_of_people = len(people)
    number_of_links = range(len(people[0].links))
    if number_of_people ==2:
        link_mae =[]
        for link_id in number_of_links:
            link_mae.append(abs(people[0].angles()[link_id]- people[1].angles()[link_id]))
    else:
        #Each row: person column: link_id
        angle_list = [people[x].angles() for x in range(number_of_people)]
        stacked_angles= np.vstack(angle_list)
        #Calculate mean of each link_id
        mu = np.mean(stacked_angles,axis=0)
        #Calculate errors
        errors = abs(stacked_angles - mu)
        #Calculate mean absolute error
        link_mae = np.mean(errors, axis =0)

    frame_score = np.mean(link_mae)

    return link_mae , frame_score
