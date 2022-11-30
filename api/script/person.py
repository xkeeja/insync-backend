"""
module for visualisation
"""
import numpy as np

joint_def={
    0: ["nose", "", "black", False],
    1: ["eye", "left", "green", True],
    2: ["eye", "right", "red", True],
    3: ["ear", "left", "green", True],
    4: ["ear", "right", "red", True],
    5 : ["shoulder", "left", "green", False],
    6 : ["shoulder", "right", "red", False],
    7 : ["elbow", "left", "green", False],
    8 : ["elbow", "right", "red", False],
    9 : ["wrist", "left", "green", False],
    10 : ["wrist", "right", "red", False],
    11 : ["hip", "left", "green", False],
    12 :["hip", "right", "red", False],
    13 : ["knee", "left", "green", False],
    14 :["knee", "right", "red", False],
    15 : ["foot", "left", "green", False],
    16: ["foot", "right", "red", False]
}

class Joint:

    def __init__(self, id :int, person_id : int):
        self.person_id = person_id
        self.id = id
        self.name = " ".join(joint_def[id][0:2])
        self.color = joint_def[id][2]
        self.is_ignored = joint_def[id][3]
        self.x = None # intented to be a float x coordinate
        self.y = None # intented to be a float y coordinate

    def add_coord(self, x : float, y : float):
        '''
        Inputs : coordinate x, y of the joint after detection
        outputs : joint itslef with coordinates updated
        '''
        self.x = x
        self.y = y
        return self

    def add_confidence(self, confidence : float):
        '''
        Inputs : confidence score from pose detection model for this joint
        outputs : joint itself with confidence scores updated
        '''
        # if confidence <= 0.2:
        #     self.bad_confidence = True
        # else:
        #     self.bad_confidence = False
        self.confidence = confidence
        return self



x = Joint(1, 3)

link_def= {
    0 : ["right ear to right eye", (4,2),"", False],
    1 : ["right eye to nose", (2,0), "", False],
    2 : ["nose to left eye", (0,1), "", False],
    3 : ["left eye to left ear", (1,3), "", False],
    4 : ["left shoulder to right shoulder", (5,6), "chest line", True],
    5 : ["right elbow to right shoulder ", (6,8), "right arm", True],
    6 : ["right elbow to right wrist", (8,10), "right forearm", True],
    7 : ["left shoulder to left elbow", (5,7), "left arm", True],
    8 : ["left elbow to left wrist", (7,9), "left forearm", True],
    9 : ["right shoulder to right hip", (6,12), "suspender right", True],
    10 : ["left shoulder to left hip", (5,11), "suspender left", True],
    11 : ["left hip to right hip", (11,12), "belt", True],
    12 : ["right hip to right knee", (12,14), "right thigh", True],
    13 : ["right knee to right foot", (14,16), "right calf", True],
    14 : ["left hip to left knee", (11,13), "left thigh", True],
    15 : ["left knee to left foot", (13,15), "left calf", True],
    16 : ["nose to left shoulder", (0,5), "nose left", True],
    17 : ["nose to right shoulder", (0,6), "nose right", True]
    }


class Link:

    def __init__(self, id:int, person_id :int):
        self.id = id
        self.person_id = person_id
        self.joint1_id = link_def[id][1][0]
        self.joint2_id = link_def[id][1][1]
        self.name = link_def[id][2]
        self.color = "yellow" # by default the color is yellow


    def add_joints(self, joint1, joint2):
        self.joints = (joint1, joint2)
        return self


    def set_color(self, color):
        self.color = color
        return self

    def add_score(self, similarity_score :float):
        self.similarity_score = similarity_score
        return self

    def add_angle(self, angle: float):
        self.angle = angle
        return self



class Person:

    def __init__(self, id:int, face_ignored :bool):
        self.id = id
        self.face_ignored = face_ignored
        self.joints = [Joint(k, id) for k in range(17)]

    def update_joints(self, x_vect, y_vect, conf_vect):
        self.joints = [joint.add_coord(x,y).add_confidence(confidence) \
            for joint ,x, y, confidence in zip(self.joints, x_vect, y_vect, conf_vect)]
        return self

    def create_links(self):
        if self.face_ignored:
            self.links_empty=[]
            for key, val in link_def.items():
                if val[3]: #filter for face mode off
                   self.links_empty.append(Link(key, self.id))

        else:
            self.links_empty = [Link(k, self.id) for k in range(16)]
        self.links = [link.add_joints(
            self.joints[link.joint1_id],
            self.joints[link.joint2_id]
            ) for link in self.links_empty
        ]
        return self

    def angles(self):
        return [link.angle for link in self.links]

    def joints_to_not_be_displayed(self):
        if self.face_ignored:
            return [joint.is_ignored for joint in self.joints ]
        else:
            return [False for _ in range(17)]

    def min_confidence(self):
        if self.face_ignored:
            list_confidence = []
            for joint in self.joints:
                if joint.is_ignored == False:
                    list_confidence.append(joint.confidence)
            return np.min(list_confidence)
        else:
            return np.min([joint.confidence for joint in self.joints])
