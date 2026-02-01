import os
import re
import numpy as np
import scipy.spatial.transform as transform
import tqdm
import open3d as o3d
import laspy

import coords

CHUNKS = 200

def load_bin(path_bin, max_distance = None, sample_ratio = None):
    assert os.path.exists(path_bin)
    if sample_ratio is not None:
        assert sample_ratio >= 0.0 and sample_ratio <= 1.0 

    print(f"Loading binary: {os.path.basename(path_bin)}")

    pts = None
    with open(path_bin, 'rb') as f:
        raw = np.frombuffer(f.read()[10240:], dtype = np.dtype([('time', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f8')]))
        if len(raw) == 0: 
            raise RuntimeError(f"Failed to parse binary file: {path_bin}")

        pts = np.stack([raw['time'], raw['x'], raw['y'], raw['z']], axis = 1)

    if pts is None: 
        raise RuntimeError(f"Failed to load binary (missing data): {os.path.basename(path_bin)}")

    print(f"Loaded {pts.shape[0]} points from binary: {os.path.basename(path_bin)}")

    if max_distance is not None:
        pts_count = pts.shape[0]
        pts_mask = np.linalg.norm(pts[:, 1:], axis = 1) <= max_distance
        pts = pts[pts_mask]
        print(f"Removed {pts_count - pts.shape[0]} points due to cutoff ({max_distance}m)")

    if sample_ratio is not None:
        pts_mask = np.random.choice(pts.shape[0], size = int(pts.shape[0] * sample_ratio), replace = False)
        pts = pts[pts_mask]
        print(f"Downsampled to {pts.shape[0]} points (ratio: {sample_ratio})")

    return pts

class Trajectory:
    def __init__(self, path_traj_raw, path_out = None, 
        # the ellipsoid used to convert to UTM
        ellipsoid = coords.ETRS89_DE,
        # offset in seconds s.t. t_scan = t_traj - offset
        traj_to_bin_offset = 0.0, 
        # whether or not to write trajectory to a KML file
        traj_export_as_kml = False,
        # 1 = best, 6 = worst
        quality_threshold = 2,
        # imu to antenna
        T_AI = np.zeros(3),
        # body to imu 
        R_IB = transform.Rotation.identity().as_matrix(), 
        # sensor to body
        R_BS = transform.Rotation.identity().as_matrix(), T_BS = np.zeros(3)):

        assert os.path.exists(path_traj_raw)
        assert path_traj_raw[-4:] == ".txt"
        assert quality_threshold >= 1 and quality_threshold <= 6

        self.T_AI, self.R_IB, self.R_BS, self.T_BS = T_AI, R_IB, R_BS, T_BS

        self.path_out = os.path.dirname(path_traj_raw) if path_out is None else path_out
        self.name = os.path.basename(path_traj_raw)[:-4]

        self.ellipsoid = ellipsoid

        self.index = 0

        def load_traj_raw():
            print(f"Loading raw trajectory: {os.path.basename(path_traj_raw)}")

            traj_raw = []
            with open(path_traj_raw, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line[0].isdigit() or ':' in line: continue
                    cols = re.split(r'\s+', line)
                    if len(cols) != 19: 
                        print(f"Skipping entry {len(traj_raw)}. Incorrect number of columns")
                        continue
                    try:
                        count += 1
                        if int(cols[11]) > quality_threshold: continue
                        traj_raw.append([
                            np.float64(cols[0]), # time
                            np.float64(cols[1]) + np.float64(cols[2]) / 60.0 + np.float64(cols[3]) / 3600.0, # lat
                            np.float64(cols[4]) + np.float64(cols[5]) / 60.0 + np.float64(cols[6]) / 3600.0, # lon
                            np.float64(cols[7]), # ellipsoidal height
                            np.float64(cols[8]), # 4 pitch
                            np.float64(cols[9]), # 5 roll 
                            np.float64(cols[10]) # 6 yaw
                        ])
                    except:
                        continue
            
            if len(traj_raw) == 0:
                raise RuntimeError(f"Failed to load trajectory: {os.path.basename(path_traj_raw)}")
            
            traj_raw = np.vstack(traj_raw)
            # ensure trajectory is sorted by timestamp before returning
            traj_raw = traj_raw[traj_raw[:, 0].argsort()]

            print(f"Loaded {len(traj_raw)} poses across {traj_raw[-1, 0] - traj_raw[0, 0]} seconds")

            return traj_raw

        file_traj = os.path.basename(path_traj_raw)[:-4] + ".npy"
        path_traj = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_traj)

        self.traj, traj_raw = None, None
        if not os.path.exists(path_traj):
            traj_raw = load_traj_raw()
            self.traj = np.copy(traj_raw)
            for i in tqdm.tqdm(range(0, len(traj_raw)), desc = "Converting ETRS89 to UTM"):
                self.traj[i, 1:4] = self.ellipsoid.ellipsoidal_to_utm(traj_raw[i, [2, 1, 3]])

            self.traj[:, 0] = traj_raw[:, 0] - traj_to_bin_offset
            self.traj[:, 4:7] = np.deg2rad(self.traj[:, 4:7])

            print(f"Saving UTM trajectory: {file_traj}");
            np.save(path_traj, self.traj)
            print(f"Finished saving UTM trajectory: {file_traj}")
        
        else:
            self.traj = np.load(path_traj)
            print(f"Loaded {self.traj.shape[0]} poses from preprocessed UTM trajectory: {file_traj}")

        if self.traj is None:
            raise RuntimeError(f"Failed to load trajectory: {os.path.basename(path_traj_raw)}")

        if traj_export_as_kml:
            if traj_raw is None:
                traj_raw = load_traj_raw()

            file_kml = self.name + ".kml"
            path_kml = os.path.join(self.path_out, file_kml)
            coords.export_as_kml(path_kml, traj_raw[:, [2, 1, 3]])

            print(f"Exported trajectory: {file_kml}")

        self.slerps = []
        for i in tqdm.tqdm(range(1, self.traj.shape[0]), desc = "Precomputing SLERPs"):
            rpy0 = self.traj[i - 1, [5, 4, 6]]
            rpy1 = self.traj[i + 0, [5, 4, 6]] 
            
            rpy0[2] = (np.pi / 2.0) - rpy0[2]
            rpy1[2] = (np.pi / 2.0) - rpy1[2]

            self.slerps.append(
                transform.Slerp([0.0, 1.0], 
                transform.Rotation.concatenate([
                    transform.Rotation.from_euler('xyz', rpy0),
                    transform.Rotation.from_euler('xyz', rpy1)])))

    def process_points(self, pts, out_format = "ply"):
        print(f"{pts.shape[0]}")
        assert pts.shape[0] > 0 and pts.shape[1] == 4
        OUT_FORMAT_OPTIONS = ["ply", "las"]
        assert out_format in OUT_FORMAT_OPTIONS

        ts, pts_s = pts[:, 0], pts[:, 1:]

        traj_idx = np.searchsorted(self.traj[:, 0], ts)
        traj_idx_mask = (traj_idx > 0) & (traj_idx < traj_idx.shape[0])
        ts = ts[traj_idx_mask]
        pts_s = pts_s[traj_idx_mask]
        traj_idx = traj_idx[traj_idx_mask]
        
        t_fst = self.traj[traj_idx - 1, 0]
        t_snd = self.traj[traj_idx, 0]
        t = (ts - t_fst) / (t_snd - t_fst)

        traj_fst = self.traj[traj_idx - 1, :]
        traj_snd = self.traj[traj_idx, :]

        t_wa = traj_fst[:, 1:4] + (traj_snd[:, 1:4] - traj_fst[:, 1:4]) * t[:, None]

        R_wa = np.empty((ts.size, 3, 3))
        for i in tqdm.tqdm(np.unique(traj_idx), desc = "Sampling SLERPs"):
            t_mask = (traj_idx == i)
            temp = self.slerps[i - 1](t[t_mask]).as_matrix()

            R_wa[t_mask] = temp

        pcd = o3d.geometry.PointCloud()
        for i, (t_wa_i, R_wa_i, pts_s_i) in tqdm.tqdm(enumerate(zip(
            np.array_split(t_wa, CHUNKS),
            np.array_split(R_wa, CHUNKS),
            np.array_split(pts_s, CHUNKS))), 
            total = CHUNKS, desc = "Processing"):

            # sensor to body
            pts_b_i = (self.R_BS @ pts_s_i.T).T + self.T_BS
            # body to imu
            pts_i_i = (self.R_IB @ pts_b_i.T).T
            # imu to antenna
            pts_a_i = pts_i_i + self.T_AI
            # antenna to world
            pts_w_i = np.matmul(R_wa_i, pts_a_i[:, :, np.newaxis])[:, :, 0] + t_wa_i    

            pcd.points.extend(pts_w_i)

        assert pts.shape[0] == len(pcd.points)

        file_out = "out_" + self.name + f"_{self.index}"
        self.index += 1
        if out_format == "las":
            file_out += ".las"
            path_out = os.path.join(self.path_out, file_out)
            las_pts = np.asarray(pcd.points)
            las_header = laspy.LasHeader(point_format = 0, version = "1.2")
            las = laspy.LasData(las_header)
            
            las.y = las_pts[:, 1]
            las.z = las_pts[:, 2]
            las.write(path_out)
        elif out_format == "ply":
            file_out += ".ply"
            path_out = os.path.join(self.path_out, file_out)
            o3d.io.write_point_cloud(path_out, pcd)

        print(f"Wrote {len(pcd.points)} points to {file_out}")
        
        return pcd