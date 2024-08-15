import numpy as np
import time
import os
import multiprocessing
import torch
# import pandas

def user_id_to_idx(inters: np.array):
    inter_list = []
    userToidx = {}
    user_num = 0
    for i in range(len(inters)):
        if inters[i][0] in userToidx:
            userToidx[inters[i][0]] = userToidx[inters[i][0]]
        else:
            userToidx[inters[i][0]] = user_num
            user_num += 1
        if inters[i][1] in userToidx:
            userToidx[inters[i][1]] = userToidx[inters[i][1]]
        else:
            userToidx[inters[i][1]] = user_num
            user_num += 1
        inter_list.append(
            torch.tensor([userToidx[inters[i][0]], userToidx[inters[i][1]]]))
    return torch.stack(inter_list), userToidx

def rate_id_to_idx(write_inters: np.array, user_user_dict: dict):
    # write_inters: userID, objectID
    write_list = []
    writeToidx, objectToidx = {}, {}
    if user_user_dict:
        write_user_num = np.max(list(user_user_dict.values())) + 1
    else:
        write_user_num = 1
    write_object_num = 0
    for i in range(len(write_inters)):
        if write_inters[i][0] in user_user_dict:
            writeToidx[write_inters[i][0]] = user_user_dict[write_inters[i][0]]
        else:
            writeToidx[write_inters[i][0]] = write_user_num
            user_user_dict[write_inters[i][0]] = write_user_num
            write_user_num += 1
        if write_inters[i][1] in objectToidx:
            objectToidx[write_inters[i][1]] = objectToidx[write_inters[i][1]]
        else:
            objectToidx[write_inters[i][1]] = write_object_num
            write_object_num += 1
        write_list.append(
            torch.tensor([writeToidx[write_inters[i][0]],objectToidx[write_inters[i][1]]]))

    return torch.stack(write_list), user_user_dict, objectToidx, write_user_num


def error_callback(error):
    print(f"Error info: {error}")

def processDataset(dataset):
    np.random.seed(2020)
    pool = multiprocessing.Pool(2)
    if dataset == 'digg_all':
        voteAddress = 'mydata/ordered_data/votes.csv'
        friendsAddress = 'mydata/ordered_data/friends.csv'

        friends, votes = np.loadtxt(friendsAddress, dtype=int), np.loadtxt(voteAddress, dtype=int)
        len_u = np.size(friends, 0)
        len_v = np.size(votes, 0)
        data_index = 0
        snap_size = 15000
        friends = friends[int(0.1 * data_index * len_u):int(0.1 * data_index * len_u) + snap_size, :]
        votes = votes[int(0.1 * data_index * len_v):int(0.1 * data_index * len_v) + snap_size, :]

    elif dataset == 'amazon':
        voteAddress = 'mydata/ordered_data/Amazon_data.csv'
        votes = np.loadtxt(voteAddress, dtype=int, comments='%')
        votes = votes[votes[:, 3].argsort()]
        len_v = np.size(votes, 0)
        data_index = 10
        snap_size = 40000
        votes = votes[int(0.1 * data_index * len_v) - snap_size:int(0.1 * data_index * len_v), :]

    elif dataset == 'yelp_hotel_labelled':
        voteAddress = 'mydata/ordered_data/yelp_hotel_labelled.csv'
        votes = np.genfromtxt(voteAddress, delimiter=',', skip_header=1)
        votes = votes[votes[:, 3].argsort()]
        len_v = np.size(votes, 0)
        data_index = 10
        snap_size = 30000
        votes = votes[int(0.1 * data_index * len_v) - snap_size:int(0.1 * data_index * len_v), :]


    if dataset == 'digg_all':
        dataRaw = np.zeros([np.size(votes, 0)+np.size(friends, 0), np.size(votes, 1)+np.size(friends, 1)-2], dtype=np.int64)

        inters_user, user_userToidx = user_id_to_idx(friends[:, [2, 3]])
        inters_rate, rate_userToidx, rate_objectToidx, write_user_num = rate_id_to_idx(votes[:, [1, 2]], user_userToidx)

        start_time = time.time()

        dataRaw[0:np.size(votes, 0), 0] = inters_rate.numpy()[0:np.size(votes, 0), 0]     #起点
        dataRaw[0:np.size(votes, 0), 1] = inters_rate.numpy()[0:np.size(votes, 0), 1] + write_user_num    #终点
        dataRaw[0:np.size(votes, 0), 3] = votes[0:np.size(votes, 0), 0]     #时间戳
        dataRaw[0:np.size(votes, 0), 2] = 0

        dataRaw[np.size(votes, 0):np.size(votes, 0)+np.size(friends, 0), 0] = inters_user.numpy()[0:np.size(friends, 0), 0]   #起点
        dataRaw[np.size(votes, 0):np.size(votes, 0)+np.size(friends, 0), 1] = inters_user.numpy()[0:np.size(friends, 0), 1]   #终点
        dataRaw[np.size(votes, 0):np.size(votes, 0)+np.size(friends, 0), 3] = friends[0:np.size(friends, 0), 1]   #时间戳
        dataRaw[np.size(votes, 0):np.size(votes, 0)+np.size(friends, 0), 4] = friends[0:np.size(friends, 0), 0]     #rating
        dataRaw[np.size(votes, 0):np.size(votes, 0)+np.size(friends, 0), 2] = 1

        dataRaw = dataRaw[dataRaw[:,3].argsort()]

        directory = "mydata"
        file_all_name = "digg_all_0.csv"

    elif dataset == 'amazon':
        dataRaw = np.zeros([np.size(votes, 0), 5],
                           dtype=np.int64)

        user_userToidx = {}
        inters_rate, rate_userToidx, rate_objectToidx, write_user_num = rate_id_to_idx(votes[:, [0, 1]], user_userToidx)

        start_time = time.time()

        dataRaw[0:np.size(votes, 0), 0] = inters_rate.numpy()[0:np.size(votes, 0), 0]  # 起点
        dataRaw[0:np.size(votes, 0), 1] = inters_rate.numpy()[0:np.size(votes, 0), 1] + write_user_num  # 终点
        dataRaw[0:np.size(votes, 0), 3] = votes[0:np.size(votes, 0), 3]  # 时间戳
        dataRaw[0:np.size(votes, 0), 2] = 0 #rate
        dataRaw[0:np.size(votes, 0), 4] = votes[0:np.size(votes, 0), 2]


        dataRaw = dataRaw[dataRaw[:, 3].argsort()]

        directory = "mydata"
        file_all_name = f"amazon_0.csv"

    elif dataset == 'yelp_hotel_labelled':
        dataRaw = np.zeros([np.size(votes, 0), 6],
                           dtype=np.int64)

        user_userToidx = {}
        inters_rate, rate_userToidx, rate_objectToidx, write_user_num = rate_id_to_idx(votes[:, [0, 1]], user_userToidx)

        start_time = time.time()

        dataRaw[0:np.size(votes, 0), 0] = inters_rate.numpy()[0:np.size(votes, 0), 0]  # 起点
        dataRaw[0:np.size(votes, 0), 1] = inters_rate.numpy()[0:np.size(votes, 0), 1] + write_user_num  # 终点
        dataRaw[0:np.size(votes, 0), 3] = votes[0:np.size(votes, 0), 3]  # 时间戳
        dataRaw[0:np.size(votes, 0), 2] = 0  # 表示类型为rate
        dataRaw[0:np.size(votes, 0), 4] = votes[0:np.size(votes, 0), 2]
        dataRaw[0:np.size(votes, 0), 5] = votes[0:np.size(votes, 0), 4]

        dataRaw = dataRaw[dataRaw[:, 3].argsort()]
        directory = "mydata"
        file_all_name = f"yelp_0_1.csv"




    data_all_path = os.path.join(directory, file_all_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(data_all_path, dataRaw, delimiter=' ', fmt='%d')


dataset = 'yelp_rest_filtered'
processDataset(dataset)


