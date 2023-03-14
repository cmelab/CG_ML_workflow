import h5py
import numpy as np
import pandas as pd
import rowan
import os

if __name__ == '__main__':
    data_path = "/home/marjan/Documents/dataset/pps_rigid/2023-03-09-18:47:47/"
    rigid_traj_path = data_path + 'rigid_traj.h5'
    f = h5py.File(rigid_traj_path, 'r')
    traj_df = pd.DataFrame()
    for frame in f.keys():
        dset = dict(f[frame])
        dset = {key: np.array(value) for key, value in dset.items()}
        dset['timestep'] = f[frame].attrs['timestep']
        traj_df = traj_df.append(dset, ignore_index=True)

    f.close()

    positions = np.asarray(traj_df["position"].tolist())
    orientations = np.asarray(traj_df["orientation"].tolist())
    net_force = np.asarray(traj_df["net_force"].tolist())
    net_torque = np.asarray(traj_df['net_torque'].tolist())


    num_particles = positions.shape[1]
    rel_positions = []
    rel_orientations = []
    for frame_positions, frame_orientations in zip(positions, orientations):
        for i in range(num_particles):
            other_particles_idx = list(range(num_particles))
            other_particles_idx.remove(i)
            rel_positions.append(frame_positions[i] - frame_positions[other_particles_idx])

            rel_orientations.append(rowan.multiply(frame_orientations[other_particles_idx],
                                                   rowan.conjugate(frame_orientations[i])))

    rel_positions = np.asarray(rel_positions)
    rel_orientations = np.asarray(rel_orientations)
    net_force = net_force.reshape((net_force.shape[0]*net_force.shape[1], net_force.shape[2]))
    net_torque = net_force.reshape((net_torque.shape[0] * net_torque.shape[1], net_torque.shape[2]))
    columns = [
        "position",
        "orientation",
        "net_force",
        "net_torque",
    ]

    new_traj_df = pd.DataFrame(columns=columns)
    new_traj_df["position"] = rel_positions.tolist()
    new_traj_df["orientation"] = rel_orientations.tolist()
    new_traj_df["net_force"] = net_force.tolist()
    new_traj_df["net_torque"] = net_torque.tolist()

    # shuffle order of frames
    new_traj_df = new_traj_df.sample(frac=1).reset_index(drop=True)

    test_frac = 0.1
    val_frac = 0.1

    dataset_len = new_traj_df.shape[0]
    test_len = int(dataset_len * test_frac)
    val_len = int(dataset_len * val_frac)

    test_df = new_traj_df.loc[:test_len]

    val_df = new_traj_df.loc[test_len: test_len + val_len]

    train_df = new_traj_df.loc[test_len + val_len:]

    target_data_path = os.path.join(data_path, 'dataset')

    if not os.path.exists(target_data_path):
        os.mkdir(target_data_path)

    train_df.to_pickle(os.path.join(target_data_path, 'train.pkl'))
    val_df.to_pickle(os.path.join(target_data_path, 'val.pkl'))
    test_df.to_pickle(os.path.join(target_data_path, 'test.pkl'))

    print('done')
