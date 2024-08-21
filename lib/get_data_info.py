from lib.preprocessing import *

# def run(test_data_list, HyperParams):
#
# 	for test_dataset in test_data_list:
# 		graphs_dataset(test_dataset, HyperParams, print_info=True)

# def get_three_pts_idx(data_idx):
# 	if data_idx == 4:
# 		return [1023, 1015, 1099] # [0.4, 0.15], [0.4, 0.2], [0.4, 0.25]
# 	elif data_idx == 6:
# 		return [1061, 996, 1000] # [0.4, 0.15], [0.4, 0.2], [0.4, 0.25]
# 	elif data_idx == 8:
# 		return [432, 434, 419] # [0.4, 0.15], [0.4, 0.2], [0.4, 0.25]
# 	elif data_idx == 10:
# 		return [523, 552, 559] # [0.4, 0.15], [0.4, 0.2], [0.4, 0.25]
# 	elif data_idx == 15:
# 		return [918, 914, 885] # [0.6, 0.15], [0.6, 0.2], [0.6, 0.25]
# 	elif data_idx == 16:
# 		return [664, 666, 653] # [0.6, 0.15], [0.6, 0.2], [0.6, 0.25]

def get_prove_pts_idx(dataset, x=0.6):

    xx = torch.tensor(dataset["mesh_pos"][0,:,0]) # shape=(1896)
    yy = torch.tensor(dataset["mesh_pos"][0,:,1]) # shape=(1896)
    xyz = [xx, yy]

    if dataset["mesh_pos"].shape[2] == 3:
       zz = dataset["mesh_pos"][0,:,2]
       xyz.append(zz)
    vel_x = torch.tensor(dataset['velocity'][:,:,0])  # shape=(600, 1896)

    targets = [torch.tensor([x, 0.05]), torch.tensor([x, 0.10]), torch.tensor([x, 0.15]),
               torch.tensor([x, 0.20]), torch.tensor([x, 0.25]), torch.tensor([x, 0.30]),
               torch.tensor([x, 0.35])]
    pts_idx_list = get_closest_point(targets, xx, yy, vel_x)

    return pts_idx_list


def get_closest_point(targets, xx, yy, vel_x):
    pts_idx_list = []
    for target in targets:
        points = torch.stack((xx, yy), dim=1)
        distances = torch.norm(points - target, dim=1)
        min_dist_index = torch.argmin(distances)
        pts_idx_list.append(min_dist_index)
        closest_point = points[min_dist_index]
        snapshot = 0
        # print("Total Node # / target node idx:", vel_x.shape[1], min_dist_index)
        # print("Closest point:", closest_point)
        # print("Distance to target:", distances[min_dist_index])
        # print("x-velocity:", vel_x[snapshot, min_dist_index])
    return pts_idx_list
