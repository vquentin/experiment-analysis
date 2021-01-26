import glob
from semimage import SEMImage 

#Load example file

example_files = glob.glob(r"./examples/*.tif")

#example_file = glob.glob(r"./examples/WF67_1_06.tif")[0]
#example_file = glob.glob(r"./examples/UB19_14.tif")[0]
#example_file = glob.glob(r"./examples/ADB1_28.tif")[0]
#example_file = glob.glob(r"./examples/WF56_1_20.tif")[0]
#example_file = glob.glob(r"./examples/ADB1_44.tif")[0]

for example_file in example_files:
    a = SEMImage(example_file, debug=True)
    
