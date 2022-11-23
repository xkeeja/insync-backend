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

    def add_coord(self, x : float, y : float):
        self.x = x
        self.y = y
        return self

    def add_confidence(self, confidence : float):
        self.confidence = confidence
        return self



x = Joint(1, 3)

class Link:

    link_def= {"left eye to left ear":(2,4),
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
