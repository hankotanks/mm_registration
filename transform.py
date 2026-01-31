
T_AI = np.array([-0.183, 0.0, 0.339]) # np.array([0.0, 0.0, 0.0]) # 

R_IB = R.identity().as_matrix() # R.from_euler('z', 90.0, degrees = True).as_matrix() # 

R_BS = R.from_euler('xyz', [0.1141, 29.5157, -0.1194], degrees = True).as_matrix() # R.identity().as_matrix() # 
T_BS = np.array([0.0, 0.0, 0.0]) # np.array([-0.7848, -0.0356, 0.2008]) # 

slerps = []
for i in tqdm.tqdm(range(1, traj.shape[0]), desc = "Precomputing SLERPs"):
    rpy0 = traj[i - 1, [5, 4, 6]]
    rpy1 = traj[i + 0, [5, 4, 6]] 
    
    rpy0[2] = (np.pi / 2.0) - rpy0[2]
    rpy1[2] = (np.pi / 2.0) - rpy1[2]

    R0 = R.from_euler('xyz', rpy0)
    R1 = R.from_euler('xyz', rpy1)

    slerps.append(Slerp([0.0, 1.0], R.concatenate([R0, R1])))

def load_and_process_bin_vectorized(path_bin):
    global traj

    file_bin = os.path.basename(path_bin)

    ts, pts_s = load_bin(path_bin)
    if DISTANCE_THRESHOLD is not None:
        pts_count = ts.size
        pts_mask = np.linalg.norm(pts_s, axis = 1) <= DISTANCE_THRESHOLD
        ts = ts[pts_mask]
        pts_s = pts_s[pts_mask]
        print(f"Removed {pts_count - ts.size} points due to cutoff ({DISTANCE_THRESHOLD}m)")

    if SAMPLE_RATIO is not None:
        pts_mask = np.random.choice(ts.size, size = int(ts.size * SAMPLE_RATIO), replace = False)
        ts = ts[pts_mask]
        pts_s = pts_s[pts_mask]
        print(f"Downsampled to {ts.size} points (ratio: {SAMPLE_RATIO})")

    traj_idx = np.searchsorted(traj[:, 0], ts)
    traj_idx_mask = (traj_idx > 0) & (traj_idx < traj_idx.shape[0])
    ts = ts[traj_idx_mask]
    pts_s = pts_s[traj_idx_mask]
    traj_idx = traj_idx[traj_idx_mask]
    
    t_fst = traj[traj_idx - 1, 0]
    t_snd = traj[traj_idx, 0]
    t = (ts - t_fst) / (t_snd - t_fst)

    traj_fst = traj[traj_idx - 1, :]
    traj_snd = traj[traj_idx, :]

    t_wa = traj_fst[:, 1:4] + (traj_snd[:, 1:4] - traj_fst[:, 1:4]) * t[:, None]

    R_wa = np.empty((ts.size, 3, 3))
    for i in tqdm.tqdm(np.unique(traj_idx), desc = "Sampling SLERPs"):
        t_mask = (traj_idx == i)
        temp = slerps[i - 1](t[t_mask]).as_matrix()

        R_wa[t_mask] = temp

    pcd = o3d.geometry.PointCloud()
    for i, (t_wa_i, R_wa_i, pts_s_i) in tqdm.tqdm(enumerate(zip(
        np.array_split(t_wa, CHUNKS),
        np.array_split(R_wa, CHUNKS),
        np.array_split(pts_s, CHUNKS))), 
        total = CHUNKS, desc = f"Processing {file_bin}"):

        # sensor to body
        pts_b_i = (R_BS @ pts_s_i.T).T + T_BS
        # body to imu
        pts_i_i = (R_IB @ pts_b_i.T).T
        # imu to antenna
        pts_a_i = pts_i_i + T_AI
        # antenna to world
        pts_w_i = np.matmul(R_wa_i, pts_a_i[:, :, np.newaxis])[:, :, 0] + t_wa_i    

        pcd.points.extend(pts_w_i)

    if SAVE_CLOUD:
        file_out = "out_" + file_bin[:-4]
        if USE_LAS_FOR_QGIS_IMPORT:
            file_out = file_out + ".las"
            path_out = os.path.join(os.path.dirname(path_bin), file_out)
            las_pts = np.asarray(pcd.points)
            las_header = laspy.LasHeader(point_format = 0, version = "1.2")
            las = laspy.LasData(las_header)
            las.x = las_pts[:, 0] - 32_000_000.0
            las.y = las_pts[:, 1]
            las.z = las_pts[:, 2]
            las.write(path_out)
        else:
            file_out = file_out + ".ply"
            path_out = os.path.join(os.path.dirname(path_bin), file_out)
            o3d.io.write_point_cloud(path_out, pcd)

        print(f"Wrote {len(pcd.points)} points to {file_out}")

    return pcd