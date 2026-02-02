import os
import sys
import numpy as np
import glob
import scipy.spatial.transform as transform
import argparse

import dataloader

parser = argparse.ArgumentParser()

parser.add_argument("path_traj",
    type = str,
    help="path to a .txt file containing the wagon trajectory")
parser.add_argument("path_scans",
    type = str,
    help = "path to a .bin or a directory of binaries containing point data")
parser.add_argument("-o", "--output",
    type = str,
    metavar = "path_out",
    help = "path to directory where point cloud files will be emitted")

args = parser.parse_args()

if not os.path.exists(args.path_traj):
    print(f"<path_traj> did not exist: {args.path_traj}")
    exit(1) 

if not os.path.exists(args.path_scans):
    print(f"<path_scans> did not exist: {args.path_scans}")
    exit(1)

if args.output is not None and not os.path.exists(args.output):
    print(f"<path_out> did not exist: {args.output}")
    exit(1)  

# this (and the process_bin invocation) should be the only place(s) with hard-coded values
traj = dataloader.Trajectory(args.path_traj, args.output,
    traj_to_bin_offset = 86418.0, 
    traj_export_as_kml = False,
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

process_bin(sorted(glob.glob(os.path.join(args.path_scans, "*.bin"))) if os.path.isdir(args.path_scans) else [args.path_scans])

print("Exiting")
