import pickle
import glob
import numpy as np 
import os.path as osp

data_folder = 'A1'
out_dir = 'round1_bg_15_e6/'
for pkl_path in glob.glob(data_folder+'/*'):
    if 'Dashboard' in pkl_path:
        # import ipdb; ipdb.set_trace()
        with open(pkl_path, 'rb') as f:
            dashboard_data = pickle.load(f)

        rear_view_path = pkl_path.replace('Dashboard', 'Rear_view')
        with open(rear_view_path, 'rb') as f:
            rear_data = pickle.load(f)

        try:
            right_view_path = pkl_path.replace('Dashboard', 'Right_side_window')
            with open(right_view_path, 'rb') as f:
                right_data = pickle.load(f)
        except:
            right_view_path = pkl_path.replace('Dashboard', 'Rightside_window')
            with open(right_view_path, 'rb') as f:
                right_data = pickle.load(f)

        # import ipdb; ipdb.set_trace()
        min_len_data = min(dashboard_data.shape[0], rear_data.shape[0], right_data.shape[0])
        dashboard_data = dashboard_data[:min_len_data]
        rear_data = rear_data[:min_len_data]
        right_data = right_data[:min_len_data]

        try:
            concat_data = np.concatenate((dashboard_data, rear_data, right_data), axis=1)
        except:
            import ipdb; ipdb.set_trace()

        save_path = out_dir + osp.basename(pkl_path).replace('Dashboard_', '')
        with open(save_path, 'wb') as fout:
            pickle.dump(concat_data, fout)