import os
import glob

# tool for cleaning screencaps folder in Makefile
files = glob.glob(f"{os.path.abspath('.')}/api/screencaps/*.jpg")
for f in files:
    os.remove(f)
