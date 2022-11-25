import numpy as np

from posevisual.person import Joint

joint1 = Joint(1,3)
joint1.add_coord(2,4)

print(joint1.id , joint1.x)
