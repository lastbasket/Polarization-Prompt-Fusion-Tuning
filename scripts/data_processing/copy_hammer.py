import os
import cv2
from glob import glob
import shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=str, required=True)
args = parser.parse_args()

original_data_folder = args.source_folder
if not os.path.exists(original_data_folder):
    print("Source data folder ({}) does not exist! Please try again".format(original_data_folder))
    exit(1)
elif not ("hammer" in os.path.split(original_data_folder)[-1] or "hammer" in os.path.split(original_data_folder)[-2]):
    print("We assume the original data folder contains the name 'hammer', otherwise we suspect there is a risk that you entered a wrong data folder path.", \
            "If indeed you are sure that the data folder contains the HAMMER dataset, please name the folder as 'hammer', which should have been default")
    exit(1)
    
new_data_folder = original_data_folder.replace("/hammer", "/hammer_polar")

print("Will copy necessary data from ({}) to ({})".format(original_data_folder, new_data_folder))
key = input("Do you confirm this? [y/n]: ")
if key != "y":
    print("Denied.")
    exit(0)
else:
    print("Accepted, copying data.")
    
scene_list = os.listdir(original_data_folder)

# -- loop over scenes --
for scene in scene_list:
    # -- skip non-folders --
    if not os.path.isdir(os.path.join(original_data_folder, scene)):
        continue
    
    print("Processing scene {}".format(scene))
    
    # -- get full path of the polarization data of this scene --
    polarization_data_folder = os.path.join(original_data_folder, scene, 'polarization')
    instance_list = os.listdir(polarization_data_folder) # _gt, depth_d435, rgb, pol, ...
    
    # -- loop over the polarization samples --
    for instance in instance_list:
        source_path = os.path.join(polarization_data_folder, instance)
        destination_path = source_path.replace(original_data_folder, new_data_folder).replace('polarization/', '')

        destination_parent_folder = Path(destination_path).parent.absolute()
        os.makedirs(destination_parent_folder, exist_ok=True)
        
        # -- copy stuff to it --
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy(source_path, destination_path)