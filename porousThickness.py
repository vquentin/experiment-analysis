import glob
from semimage.sem_image import SEMZeissImage

from matplotlib import pyplot as plt
import numpy as np

import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s')


# Load example file
example_files = glob.glob(r"./examples/*.tif")
"""
#example_file = glob.glob(r"./examples/WF67_1_06.tif")[0]
#example_file = glob.glob(r"./examples/UB19_14.tif")[0]
#example_file = glob.glob(r"./examples/ADB1_28.tif")[0]
#example_file = glob.glob(r"./examples/WF56_1_20.tif")[0]
#example_file = glob.glob(r"./examples/ADB1_44.tif")[0]
"""

#example_files = example_files1 + glob.glob(r"C:\Users\qvovermeere\OneDrive - UCL\PoSiSTAN\Experiments\SEM WinFab\Quentin VO\ADB1\*.tif")
#example_files = glob.glob(r"C:\Users\qvovermeere\OneDrive - UCL\PoSiSTAN\Experiments\SEM WinFab\Quentin VO\ADB1\cavity\ADB1_37.tif")
#example_files = glob.glob(r"C:\Users\qvovermeere\OneDrive - UCL\PoSiSTAN\Experiments\SEM WinFab\Quentin VO\ADB1\cavity\*.tif")
#example_files = [glob.glob(r"C:\Users\qvovermeere\OneDrive - UCL\PoSiSTAN\Experiments\SEM WinFab\Quentin VO\WF56\*.tif")[14]]
#example_files = glob.glob(r"C:\Users\qvovermeere\OneDrive - UCL\PoSiSTAN\Experiments\SEM WinFab\Quentin VO\WF56\*.tif")

x = []
y = []
cavity = []
porous = []
names = []

for example_file in example_files:
    a = SEMImage(filePath=example_file, debug=True)
    try:
        x.append(a.stageX)
        y.append(a.stageY)
        names.append(a.imageName)
        cavity.append(a.cavity)
        try:
            porous.append(a.porous)
        except:
            x.pop()
            y.pop()
            cavity.pop()
            names.pop()
            print("porous does not exist")
    except:
        x.pop()
        y.pop()
        names.pop()
        print("cavity does not exist")

x = np.array(x)
y = np.array(y)
cavity_thick = [sub['thick']/1000 for sub in cavity]
cavity_thick_unc = [sub['thick_unc']/1000 for sub in cavity]
porous_thick = [sub['thick']/1000 for sub in porous]
porous_thick_unc = [sub['thick_unc']/1000 for sub in porous]

print(f"Number of detected cavities: {len(cavity_thick)}")

plt.figure()
plt.errorbar(x, cavity_thick, yerr=cavity_thick_unc, fmt='o')
plt.xlabel('Distance (mm)')
plt.ylabel('Cavity depth (um)')


plt.figure()
plt.errorbar(x, porous_thick, yerr=porous_thick_unc, fmt='o')
plt.xlabel('Distance (mm)')
plt.ylabel('Porous Si thickness (um)')

plt.show()

#a = SEMImage(filePath=example_files[10])
