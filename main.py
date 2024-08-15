import copy
import time
import torch
import numpy as np
import os
import argparse
import random
from utils import *
from torch.utils.tensorboard import SummaryWriter
from DynamicDatasetLoader import DynamicDataset_loader
from mode import Model
from Component import *
# from torchviz import make_dot
import pickle
import torch.nn.functional as F
import torch.utils.data
from prepare_data import generateDataset
from torch.cuda.amp import GradScaler, autocast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def initialize():
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    print(nowdt())
    print(args)
    if args.dataset == 'digg':
        args.dataset='digg_all'

    # tensorboard
    tb_path = f'runs/{args.dataset}/{args.data_index}_{args.train_ratio}_{args.anomaly_ratio}_{args.snap_size}/'
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb = SummaryWriter(log_dir=tb_path + f"{time.strftime('%m-%d,%H:%M:%S')}")

    # log
    log_path = f'logs/{args.dataset}/{args.data_index}_{args.train_ratio}_{args.anomaly_ratio}_{args.snap_size}_{time.time()}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"{time.strftime('%m-%d %H:%M:%S')}")
    log = open(log_file, "a")
    log.writelines(time.strftime('%m-%d %H:%M:%S') + "\n")

    # checkpoints
    checkpoints_path = f'checkpoints/{args.dataset}/{args.data_index}_{args.train_ratio}_{args.anomaly_ratio}_{args.snap_size}_{time.time()}'
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    args.checkpoints_path = os.path.join(checkpoints_path, 'checkpoint.pt')

    # Init dataloader
    dataset = {}
    if args.dataset not in ['digg_all', 'amazon', 'yelp']:
        raise NotImplementedError
    generateDataset(args.dataset, args.snap_size, train_per=args.train_ratio, anomaly_per=args.anomaly_ratio, args=args, on_test=args.onTest)
    print("0_prepare_data has done")
    dataset = DynamicDataset_loader(args= args)
    dataset = dataset.load()
    t_embedding_begin = time.time()

    GE_path = f'data/temp/{args.dataset}/'
    if not os.path.exists(GE_path):

        t_embedding_begin = time.time()
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, type_embeddings, neighbour_snaps, hsg_snaps, edges_snaps = \
            generate_embedding(dataset, args)
        print(f'generate_embedding() uses {time.time() - t_embedding_begin}')

        os.makedirs(GE_path)
        with open('data/temp/' + f'{args.dataset}/' + args.dataset + '_GE'+ '_' + str(0) + '_' + str(args.train_ratio) + '_' + str(args.anomaly_ratio) + '.pkl', 'wb')\
                as f:
            pickle.dump((raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, type_embeddings, neighbour_snaps, hsg_snaps, edges_snaps), f,
                        pickle.HIGHEST_PROTOCOL)
    else:
        with open('data/temp/' + f'{args.dataset}/' + args.dataset + '_GE' + '_' + str(0) + '_' + str(args.train_ratio) + '_' + str(args.anomaly_ratio) + '.pkl', 'rb') \
            as f:
            raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, type_embeddings, neighbour_snaps, hsg_snaps, edges_snaps = pickle.load(f)

    print(f'generate_embedding() uses {time.time() - t_embedding_begin}')

    # Start Training
    print("Now begin training!")
    max_epoch = args.epochs
    model, optimizer = initialize()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch*len(dataset['snap_train']),
                                                           eta_min=args.lr / 100)


    max_all_node_idx = np.max(dataset['idx'])

    datas_train = []
    datas_test = []
    for i in dataset['snap_train']:
        if wl_embeddings[i] is None:
            continue
        snap = {}
        snap['int_embedding'] = int_embeddings[i]
        snap['hop_embedding'] = hop_embeddings[i]
        snap['time_embedding'] = time_embeddings[i]
        snap['type_embedding'] = type_embeddings[i]
        snap['neighbour_edges'] = neighbour_snaps[i]
        snap['edges_snap'] = edges_snaps[i]
        snap['y'] = dataset['y'][i].unsqueeze(1).float()
        datas_train.append(snap)
    datas_train = torch.utils.data.DataLoader(datas_train, batch_size=1, num_workers=4,pin_memory=True)
    for i in dataset['snap_test']:
        if wl_embeddings[i] is None:
            continue
        snap = {}
        snap['int_embedding'] = int_embeddings[i]
        snap['hop_embedding'] = hop_embeddings[i]
        snap['time_embedding'] = time_embeddings[i]
        snap['type_embedding'] = type_embeddings[i]
        snap['neighbour_edges'] = neighbour_snaps[i]
        snap['edges_snap'] = edges_snaps[i]
        snap['y'] = dataset['y'][i].unsqueeze(1).float()
        datas_test.append(snap)
    datas_test = torch.utils.data.DataLoader(datas_test, batch_size=1, num_workers=4,pin_memory=True)
    time0 = time.time()


    mask_file = 'data/temp/' + f'{args.dataset}/' + args.dataset + '_MK'+ '_' + str(0) + '_' + str(args.train_ratio) + '_' + str(args.anomaly_ratio) + '.pkl'
    if not os.path.exists(mask_file):
        mask_hsg_rs_snaps = prepare_hsg_snaps(args, hsg_snaps, hop_embeddings, type_embeddings)
        combined_hsgs = prepare_combined_hsgs(args, hsg_snaps)

        with open(mask_file, 'wb')\
                as f:
            pickle.dump((mask_hsg_rs_snaps, combined_hsgs), f,
                        pickle.HIGHEST_PROTOCOL)
    else:
        with open(mask_file, 'rb') \
            as f:
            mask_hsg_rs_snaps, combined_hsgs = pickle.load(f)

    time1 =time.time()
    print(f'mask_hsg_rs:{time1 - time0}')

    step_num = 0
    max_auc = 0
    max_auc_epoch = -1
    time4train = 0
    time4test = 0
    time_begintrain = time.time()
    early_stopping = EarlyStopping(args=args, patience=args.patience, verbose=True)
    print(f'model\'s device:{next(model.parameters()).device}')
    print(f'Model #Params: {get_n_params(model)}.')
    for epoch in range(max_epoch):
        time_epoch_start = time.time()
        with torch.autograd.set_detect_anomaly(True):
            model.train()
            t_epoch_begin = time.time()

            all_z = [torch.zeros((max_all_node_idx + 1, args.hidden_size_HANLayer * args.num_heads_HANLayer)).to(
                args.device)] * (args.window_size - 1)
            all_node_idx = [None] * (args.window_size - 1)

            loss_train = 0
            reg_loss = 0
            for i, data in enumerate(datas_train):

                snap = i+args.window_size - 1
                int_embedding = torch.squeeze(data['int_embedding']).to(args.device)
                hop_embedding = torch.squeeze(data['hop_embedding']).to(args.device)
                time_embedding = torch.squeeze(data['time_embedding']).to(args.device)
                type_embedding = torch.squeeze(data['type_embedding']).to(args.device)
                neighbour_edges = torch.squeeze(data['neighbour_edges']).to(args.device)
                edges_snap = torch.squeeze(data['edges_snap']).to(args.device)
                y = data['y'][0].to(args.device)
                hsg_edges = hsg_snaps[snap]
                edges_type = dataset['edges'][snap][:, 2]
                if mask_hsg_rs_snaps is not None:
                    mask_hsg_rs = mask_hsg_rs_snaps[snap]
                    combined_graph_data = combined_hsgs[snap]
                else:
                    mask_hsg_rs = None

                if snap == 9:
                    print(1)
                time1 = time.time()

                bce_loss1, reg_loss1, nce_loss1, all_z, all_node_idx, _ = model(y, dataset['id_type_map'], int_embedding,
                                                                               hop_embedding, time_embedding,
                                                                               type_embedding,
                                                                                neighbour_edges, edges_snap,
                                                                               all_z, all_node_idx,
                                                                                copy.deepcopy(mask_hsg_rs),
                                                                                combined_graph_data,
                                                                                istest=False
                                                                                )

                bce_loss1 = bce_loss1.squeeze()
                loss = bce_loss1.mean()
                loss = loss + reg_loss1 + nce_loss1


                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                step_num += 1
                scheduler.step()
                print('epoch:{}, snap:{}, loss:{:.4f}, time{}'
                      .format(epoch, snap, loss.detach().item(), time.time() - time1))

                time_begintest = time.time()
                model.eval()
                print('test')
                preds = []
                sims = {}
                for i, data in enumerate(datas_test):
                    time1 = time.time()
                    snap = len(dataset['snap_train']) + i
                    int_embedding = torch.squeeze(data['int_embedding']).to(args.device)
                    hop_embedding = torch.squeeze(data['hop_embedding']).to(args.device)
                    time_embedding = torch.squeeze(data['time_embedding']).to(args.device)
                    type_embedding = torch.squeeze(data['type_embedding']).to(args.device)
                    neighbour_edges = torch.squeeze(data['neighbour_edges']).to(args.device)
                    edges_snap = torch.squeeze(data['edges_snap']).to(args.device)
                    y = data['y'][0].to(args.device)
                    hsg_edges = hsg_snaps[snap]
                    if mask_hsg_rs_snaps is not None:
                        mask_hsg_rs = mask_hsg_rs_snaps[snap]
                        combined_graph_data = combined_hsgs[snap]
                    else:
                        mask_hsg_rs = None

                    with torch.no_grad():
                        bce_loss1, reg_loss1, nce_loss, all_z, all_node_idx, predict = model(y, dataset['id_type_map'], int_embedding, hop_embedding,
                                                       time_embedding,
                                                       type_embedding, neighbour_edges, edges_snap, all_z,
                                                       all_node_idx, copy.deepcopy(mask_hsg_rs),combined_graph_data,
                                                        istest=True)
                    if snap == len(dataset['snap_train']):
                        bce_loss2 = bce_loss1
                        reg_loss2 = reg_loss1
                    else:
                        bce_loss2 = torch.vstack([bce_loss2, bce_loss1])
                        reg_loss2 = reg_loss2 + reg_loss1

                    pred = predict.squeeze().to('cpu')
                    preds.append(pred)
                    print('epoch:{}, snap:{}, bce_loss:{:.4f}, time{}'
                          .format(epoch, snap, bce_loss1.squeeze().mean().detach().item(), time.time() - time1))

                y_test = dataset['y'][min(dataset['snap_test']):max(dataset['snap_test']) + 1]
                y_test = [y_snap.numpy() for y_snap in y_test]

                aucs, auc_full = evaluate(snap_test=dataset['snap_test'], trues=y_test, preds=preds)
                if auc_full > max_auc:
                    max_auc = auc_full
                    max_auc_epoch = epoch


                for i in range(len(dataset['snap_test'])):
                    print("Snap: %02d | AUC: %.4f" % (dataset['snap_test'][i], aucs[i]))
                print(
                    'TOTAL AUC:{:.4f}  bce_loss:{:.4f}  MAX_AUC:{:.4f} MAX_AUC_EPOCH:{:.4f}'.format(
                        auc_full, bce_loss2.squeeze().mean().detach().item(), max_auc, max_auc_epoch))
                print(f'******* epoch:{epoch} totally uses {time.time() - time_epoch_start}s')
                time4test = time4test + time.time() - time_begintest
                early_stopping(auc_full, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    time4train = time.time() - time_begintrain - time4test


    print(f"time4train: {time4train}")
    print(f"time4test: {time4test}")
    tb.close()


