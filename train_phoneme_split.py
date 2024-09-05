import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import os
from timeit import default_timer as timer
import argparse
import numpy as np
import editdistance

from data import consts
from utils import optimizer_util
from utils.logging_util import get_one_logger
from models.phoneme_split_net import PhonemeSplitDecoderNet


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='2')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size.')
    parser.add_argument('--epoch_num', type=int, help='epoch num when training', default=60)
    parser.add_argument('--save_epochs', type=int, default=3, help='epoch interval to save checkpoint')
    parser.add_argument('--load_name', help='load pkl file name', default=None)
    parser.add_argument('--save_name', help='saved pkl file name', default='test')
    return parser.parse_args()


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def dataset_factory(dataset_name):
    if dataset_name == "LRS2":
        from data.LRS2_dataset import LRS2Dataset, collate_fn
        train_dataset = LRS2Dataset("train")
        unseen_dataset = LRS2Dataset("test")
    else:
        raise Exception("Wrong dataset_name")
    return train_dataset, unseen_dataset, collate_fn


def main():
    args = arg_parse()
    torch.cuda.set_device(int(args.gpu))
    # save_dir = "./log/" + args.save_name
    save_dir = consts.pkl_root + args.save_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_one_logger(save_dir)
    init_seeds(1, cuda_deterministic=False)
    num_class = 51+2+2  # 包括padding和SIL, bos和eos
    model = PhonemeSplitDecoderNet(num_class=num_class)
    logger.info("Num of class: " + str(num_class))
    model.cuda()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    ph_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    dataset = "LRS2"
    train_dataset, unseen_dataset, collate_fn = dataset_factory(dataset)
    logger.info("{} train data size: {}".format(dataset, len(train_dataset)))
    logger.info("{} test data size: {}".format(dataset, len(unseen_dataset)))

    warm_up_epochs = 8
    dataset_len = len(train_dataset)
    max_step = int(args.epoch_num * dataset_len / args.batch_size)
    warm_up_step = int(dataset_len / args.batch_size * warm_up_epochs)

    lr_scale = 0.5
    param_groups = [{'params': model.parameters()},
                    # {'params': loss_fn.parameters()},
                    ]
    _optimizer = torch.optim.Adam(param_groups, lr=0.001, weight_decay=1e-4, betas=(0.9, 0.997), eps=1e-9)
    model_opt = optimizer_util.NoamOpt(consts.EMBED_SIZE, lr_scale, warmup=warm_up_step, optimizer=_optimizer, max_step=max_step)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_opt.optimizer, factor=0.5, mode='max', patience=8, cooldown=16, min_lr=3e-7)

    if not os.path.exists(consts.pkl_root):
        os.makedirs(consts.pkl_root)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3,
                                   pin_memory=True, collate_fn=collate_fn)
    unseen_data_loader = DataLoader(unseen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3,
                                    pin_memory=True, collate_fn=collate_fn)

    total_time = 0
    for epoch in range(1, args.epoch_num + 1):

        start_time = timer()
        model.train()
        # loss_fn.train()
        train_losses = 0
        losses1 = 0
        losses2 = 0
        losses3 = 0
        epoch_acc = []
        ph_epoch_cer = []
        for img_sequence, frame_label, ph_label, ph_label_w_sil in train_data_loader:
            img_sequence = img_sequence.cuda()
            frame_label = frame_label.cuda()
            ph_label = ph_label.cuda()
            ph_label_w_sil = ph_label_w_sil.cuda()
            mask = (ph_label != 0)
            frame_mask = (frame_label != 0)
            frame_logits, ph_logits, bound_logits = model(img_sequence, ph_label, label_mask=mask, frame_mask=frame_mask, ph_label_w_sil=ph_label_w_sil)  # (B, T, num_class)
            batch = frame_label.shape[0]
            ph_seq_loss = ph_loss_fn(ph_logits.reshape(-1, num_class), ph_label_w_sil[:, 1:].reshape(-1))
            loss = loss_fn(frame_logits.reshape(-1, num_class), frame_label.reshape(-1))
            boundary_label = get_boundary_label(frame_label).cuda().float()
            boundary_loss = F.binary_cross_entropy_with_logits(bound_logits, boundary_label, weight=frame_mask.int())

            total_loss = loss + ph_seq_loss + boundary_loss
            model_opt.zero_grad()
            total_loss.backward()
            model_opt.step()
            preds = frame_logits.argmax(-1).cpu().numpy()
            frame_label = frame_label.cpu().numpy()

            ph_preds = ph_logits.argmax(-1).cpu().numpy()
            ph_label = ph_label_w_sil[:, 1:].cpu().numpy()
            for i in range(batch):
                ph_len = np.sum(ph_label[i] != 0)
                ph_truth = (ph_label[i][:ph_len]).tolist()
                ph_pred = (ph_preds[i][:ph_len]).tolist()
                ph_epoch_cer.append(editdistance.eval(ph_truth, ph_pred)/ph_len)

                frame_len = np.sum(frame_label[i] != 0)
                truth = frame_label[i][:frame_len]
                prediction = preds[i][:frame_len]
                epoch_acc.append(sum([1 for a, b in zip(prediction, truth.tolist()) if a == b]) / frame_len)
            train_losses += total_loss.item()
            losses1 += loss.item()
            losses2 += ph_seq_loss.item()
            losses3 += boundary_loss.item()

        ph_epoch_cer = np.mean(ph_epoch_cer)
        epoch_acc = np.mean(epoch_acc)
        test_interval = args.save_epochs
        if epoch % test_interval == 0:
            test_cer, test_acc = get_test_cer(model, unseen_data_loader, logger)
            logger.info("Test cer: {:.4f}, acc: {:.4f}".format(test_cer, test_acc))

        end_time = timer()
        total_time += end_time - start_time
        logger.info(
            f"Epoch: {epoch}, Epoch time = {(end_time - start_time):.3f}s, Total time = {(total_time / 3600.0):.2f}hour,"
            f" l_rate: {model_opt.get_rate(): .7f}")
        logger.info("Total train loss: {:.6f}".format(train_losses))
        logger.info("Frame level pred loss: {:.4f}, train acc: {:.4f}".format(losses1, epoch_acc))
        logger.info("Phoneme seq pred loss: {:.4f}, train cer: {:.4f}".format(losses2, ph_epoch_cer))
        logger.info("Frame level boundary loss: {:.4f}".format(losses3))
        logger.info("\n")
        scheduler.step(epoch_acc)

    state = {
        'state_dict': model.state_dict(),
    }
    pkl_name = args.save_name + ".pkl"
    logger.info("parameter pkl saving as " + pkl_name)
    torch.save(state, consts.pkl_root + args.save_name + "/" + pkl_name)

    logger.info("end of training")


def calc_cer(predict, truth):
    # cer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in zip(predict, truth)]
    cer = editdistance.eval(predict, truth) / len(truth)
    return cer


def get_boundary_label(frame_label):
    seq_len = frame_label.shape[1]
    batch_boundary_label = []
    for label in frame_label:
        boundary_label = [0]
        for i in range(1, seq_len):
            if label[i] != 0:
                if label[i] == label[i-1]:
                    boundary_label.append(0)
                else:
                    boundary_label.append(1)
            else:
                boundary_label.append(0)
        batch_boundary_label.append(boundary_label)
    batch_boundary_label = torch.tensor(batch_boundary_label)
    return batch_boundary_label


def get_test_cer(model, dataloader, logger=None):
    test_display = 0
    cer_list = []
    acc_list = []
    ph_cer_list = []
    bound_acc_list = []
    with torch.no_grad():
        model.eval()
        for img_sequence, frame_label, ph_label, ph_label_w_sil in dataloader:
            img_sequence = img_sequence.cuda()
            frame_label = frame_label.cuda()
            ph_label = ph_label.cuda()
            ph_label_w_sil = ph_label_w_sil.cuda()
            mask = (ph_label != 0)
            frame_mask = (frame_label != 0)
            logits, ph_preds, bound_preds = model.predict(img_sequence, ph_label, label_mask=mask, frame_mask=frame_mask)
            batch = frame_label.shape[0]
            boundary_label = get_boundary_label(frame_label).numpy()
            preds = logits.argmax(-1).cpu().numpy()
            frame_label = frame_label.cpu().numpy()
            # ph_preds = ph_logits.argmax(-1).cpu().numpy()
            ph_label_w_sil = ph_label_w_sil.cpu().numpy()
            for i in range(batch):
                ph_len = np.sum(ph_label_w_sil[i] != 0)
                ph_truth = (ph_label_w_sil[i][:ph_len][1:-1]).tolist()
                ph_pred = ph_preds[i][1:-1].cpu().numpy().tolist()
                # ph_pred = (ph_preds[i][:ph_len]).tolist()
                ph_cer_list.append(calc_cer(ph_pred, ph_truth))

                frame_len = np.sum(frame_label[i] != 0)
                bound_acc = np.sum(boundary_label[i][:frame_len] == (bound_preds[i][:frame_len] > 0.5).cpu().numpy()) / frame_len
                bound_acc_list.append(bound_acc)

                prediction = preds[i][:frame_len]
                truth = frame_label[i][:frame_len]
                acc_list.append(sum([1 for a, b in zip(prediction, truth.tolist()) if a == b]) / len(truth))
                cer_list.append(calc_cer(prediction, truth))
                if logger is not None and test_display < 3:
                    logger.info("pred : {}, ".format(prediction))
                    logger.info("truth: {}".format(truth))
                    test_display += 1
    test_cer = np.mean(cer_list)
    test_acc = np.mean(acc_list)
    logger.info("Test phoneme cer with silence: {:.4f}".format(np.mean(ph_cer_list)))
    logger.info("Test boundary predict acc: {:.4f}".format(np.mean(bound_acc_list)))
    return test_cer, test_acc


if __name__ == "__main__":
    main()
