import os
import sys
import numpy as np
import glob
import scipy.spatial.transform as transform

import dataloader

if not len(sys.argv) == 3:
    print(f"Usage: {sys.argv[0]} <path_scans> <path_traj>")
    exit(1)

path_scans = sys.argv[1]
if not os.path.exists(path_scans):
    print(f"<path_scans> did not exist: {path_scans}")
    exit(1)

path_traj_raw = sys.argv[2]
if not os.path.exists(path_traj_raw):
    print(f"<path_traj> did not exist: {path_traj_raw}")
    exit(1)

# this (and the process_bin invocation) should be the only place(s) with hard-coded values
traj = dataloader.Trajectory(path_traj_raw, 
    traj_to_bin_offset = 86418.0, 
    T_AI = np.array([-0.183, 0.0, 0.339]), 
    R_BS = transform.Rotation.from_euler('xyz', [0.1141, 29.5157, -0.1194], degrees = True).as_matrix())

def process_bin(path_bins, max_distance = None, sample_ratio = 0.01):
    for path_bin in path_bins:  
        try:
            pts = dataloader.load_bin(path_bin, max_distance, sample_ratio)
            file_bin = os.path.basename(path_bin)
            if pts.shape[0] == 0:
                print(f"Skipping {file_bin} (read 0 points)")
                continue
            print(f"Read {pts.shape[0]} points from {file_bin}")
            traj.process_points(pts)
        except Exception as e:
            print(f"{e}")

process_bin(sorted(glob.glob(os.path.join(path_scans, "*.bin"))) if os.path.isdir(path_scans) else [path_scans])
