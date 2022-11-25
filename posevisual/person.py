"""
module for visualisation

"""
joint_def={
    0: ["nose", "", "black"],
    1: ["eye", "left", "green"],
    2: ["eye", "right", "red"],
    3: ["ear", "left", "green"],
    4: ["ear", "right", "red"],
    5 : ["shoulder", "left", "green"],
    6 : ["shoulder", "right", "red"],
    7 : ["elbow", "left", "green"],
    8 : ["elbow", "right", "red"],
    9 : ["wrist", "left", "green"],
    10 : ["wrist", "right", "red"],
    11 : ["hip", "left", "green"],
    12 :["hip", "right", "red"],
    13 : ["knee", "left", "green"],
    14 :["knee", "right", "red"],
    15 : ["foot", "left", "green"],
    16: ["foot", "right", "red"]
}

class Joint:

    def __init__(self, id :int, person_id : int):
        self.person_id = person_id
        self.id = id
        self.name = " ".join(joint_def[id][0:2])
        self.color = joint_def[id][2]
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
        self.confidence = confidence
        return self



x = Joint(1, 3)

link_def= {
    0 : ["right ear to right eye", (4,2),""],
    1 : ["right eye to nose", (2,0), ""],
    2 : ["nose to left eye", (0,1), ""],
    3 : ["left eye to left ear", (1,3), ""],
    4 : ["left shoulder to right shoulder", (5,6), "chest line"],
    5 : ["right elbow to right shoulder ", (6,8), "right arm"],
    6 : ["right elbow to right wrist", (8,10), "right forearm"],
    7 : ["left shoulder to left elbow", (5,7), "left arm"],
    8 : ["left elbow to left wrist", (7,9), "left forearm"],
    9 : ["right shoulder to right hip", (6,12), "suspender right"],
    10 : ["left shoulder to left hip", (5,11), "suspender left"],
    11 : ["left hip to right hip", (11,12), "belt"],
    12 : ["right hip to right knee", (12,14), "right thigh"],
    13 : ["right knee to right foot", (14,16), "right calf"],
    14 : ["left hip to left knee", (11,13), "left thigh"],
    15 : ["left knee to left foot", (13,15), "left calf"]
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



class Person:

    def __init__(self, id:int):
        self.id = id
        self.joints = [Joint(k, id) for k in range(17)]

    def update_joints(self, x_vect, y_vect, conf_vect):
        self.joints = [joint.add_coord(x,y).add_confidence(confidence) \
            for joint ,x, y, confidence in zip(self.joints, x_vect, y_vect, conf_vect)]
        return self

    def create_links(self):
        self.links_empty = [Link(k, self.id) for k in range(16)]
        self.links = [link.add_joints(
            self.joints[link.joint1_id],
            self.joints[link.joint2_id]
            ) for link in self.links_empty]
        return self
