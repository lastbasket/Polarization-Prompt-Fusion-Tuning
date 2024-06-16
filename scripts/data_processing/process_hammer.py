import os
import numpy as np
import cv2
import math
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=str, required=True)
args = parser.parse_args()
dataroot = args.source_folder
if not os.path.exists(dataroot):
    print("Source data folder ({}) does not exist! Please copy polarization data from original hammer first!".format(dataroot))
    exit(1)
    
print("Processing data from ({})".format(dataroot))
# viewing_direction = np.load('vd.npy')

def process_polar(pol_path):
    polar_raw = cv2.imread(pol_path, -1)

    H = polar_raw.shape[0]
    W = polar_raw.shape[1]

    # The following filter angle mapping to the specific block in the png is by guess, not sure if it is the correct one, verification from the author is required
    pol_0 = polar_raw[0:H//2, 0:W//2, :] # top left - 0
    pol_45 = polar_raw[0:H//2, W//2:, :] # top right - 45
    pol_90 = polar_raw[H//2:, 0:W//2:, :] # bottom left - 90
    pol_135 = polar_raw[H//2:, W//2:, :] # bottom right - 135

    # -- convert each measurement under a specific polarizer angle to grayscale --
    pol_0, pol_45, pol_90, pol_135 = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, None].astype(np.float64) for im in [pol_0, pol_45, pol_90, pol_135]]

    pol_0 = pol_0 / 255
    pol_45 = pol_45 / 255
    pol_135 = pol_135 / 255
    pol_90 = pol_90 / 255

    # Q = (pol_0 - pol_90).astype(np.float64)
    # U = (pol_45 - pol_135).astype(np.float64)
    # Q[Q == 0] = 1e-6
    # U[U == 0] = 1e-6

    # iun = (pol_0 + pol_45 + pol_90 + pol_135) / 2
    # iun[iun == 0] = 1e-6
    
    # from -pi/2 to pi/2
    # phi = 1/2 * np.arctan((U)/(Q))

    # -pi/2, pi/2 -> 0-1
    # phi = (phi + math.pi)%math.pi
    # sign of cos(2phi) = sign of Q
    # cos_2phi = np.cos(2*phi)
    # check_sign = cos_2phi * Q
    # phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.

    # cos_2phi = np.cos(2*phi)
    # check_sign = cos_2phi * Q
    # phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
    # phi = (phi + math.pi)%math.pi

    # rho = np.sqrt((Q)**2+(U)**2) / iun
    # rho[rho>1] = 1 
    
    I = (pol_0 + pol_45 + pol_90 + pol_135) / 2.
    Q = (pol_0 - pol_90).astype(np.float64)
    U = (pol_45 - pol_135).astype(np.float64)
    Q[Q == 0] = 1e-10
    I[I == 0] = 1e-10
    rho = (np.sqrt(np.square(Q)+np.square(U))/I).clip(0,1)
    phi = 0.5 * np.arctan(U/Q)
    cos_2phi = np.cos(2*phi)
    check_sign = cos_2phi * Q
    phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
    phi = (phi + math.pi)%math.pi
    # degree of polarization Ï rho, phi is angle of pol
    # print(iun.shape, phi.shape, rho.shape, viewing_direction.shape)
    pol_reps = np.concatenate([I, rho, phi], axis=2)

    # pol_reps = np.concatenate([pol_0[...,None], pol_45[...,None], pol_90[...,None], pol_135[...,None]], axis=2)

    return pol_reps

data_list = sorted(os.listdir(dataroot))

for i in data_list:
    if not ("scene" in i and "traj" in i and (not "naked" in i)):
        continue

    idx = int(i.split('scene')[1].split('_tra')[0])
    data_path = os.path.join(dataroot, i)
    data_pol_path = os.path.join(data_path, 'pol')

    sub_list = sorted(os.listdir(data_pol_path))

    for j in sub_list:
        pol_path = os.path.join(data_pol_path, j)
        rgb_path = pol_path.replace('/pol/', '/rgb/')

        new_pol_dir = data_pol_path.replace('/pol', '/pol_processed')
        os.makedirs(new_pol_dir, exist_ok=True)

        new_pol_path = pol_path.replace('/pol/', '/pol_processed/').replace('png', 'npy')

        print(new_pol_path)

        pol_reps = process_polar(pol_path)
        # print("--> Polarization representation shape {}".format(pol_reps.shape))
        np.save(new_pol_path, pol_reps)

        # print(new_pol_path)
        # exit()