from tqdm import tqdm
import torch
from utils_training.utils import flow2kps
from utils_training.evaluation import Evaluator
from info_nce import InfoNCE
r"""
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
"""

def normlize(corr_matrix):
    corr_matrix = (corr_matrix - corr_matrix.min()) / (
        corr_matrix.max() - corr_matrix.min()
    )
    return corr_matrix

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        # ipdb.set_trace()
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum() / torch.sum(~mask)

### kp: 列，行
def train_epoch(net, optimizer, train_loader, device, epoch, train_writer):
    n_iter = epoch * len(train_loader)
    net.train()
    running_total_loss = 0
    running_total_loss_recover = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    epoch_t = epoch
    if epoch_t == 0:
        epoch_t = 1
    if epoch_t > 10:
        rate = 0.02
    else:
        rate = 0.2 / epoch_t
    print(rate)

    for i, mini_batch in pbar:
        optimizer.zero_grad()
        flow_gt_src = mini_batch["flow_src"].to(device)
        src_img = mini_batch["src_img"].to(device)
        flow, _, mask_index, src_recover = net(
            src_img,
            mini_batch["trg_img"].to(device),
            is_train=True
        )
        loss_contrust = InfoNCE(negative_mode='paired')
        Loss_contrust = 0.0
        B = flow.shape[0]
        for b in range(B):
            mask_index_b = mask_index[b]
            src_recover_b = src_recover[b]
            index_shape = mask_index_b.shape[0]
            querys = []
            positives = []
            negatives = []
            for msk_id in range(index_shape):
                querys.append(src_recover_b[mask_index_b[msk_id]])
                index = mask_index_b[msk_id]-1
                pos_token = torch.zeros_like(src_recover_b[mask_index_b[msk_id]])
                pos_token = pos_token.to(device)
                cou = 0
                index_x = index // 16
                index_y = index % 16
                for x_of in range(-1, 2):
                    for y_of in range(-1, 2):
                        if x_of == 0 and y_of == 0:
                            continue
                        select_index_x = index_x + x_of
                        select_index_y = index_y + y_of
                        if select_index_x in range(16) and select_index_y in range(16):
                            select_index = select_index_x * 16 + select_index_y + 1
                            cou += 1
                            pos_token = pos_token + src_recover_b[select_index]

                pos_token = pos_token / cou
                positives.append(pos_token)
                negative_index = torch.cat([mask_index_b[:msk_id], mask_index_b[msk_id+1:]])
                negatives.append(src_recover_b[negative_index])

            if index_shape != 0:
                querys = torch.stack(querys)
                with torch.no_grad():
                    positives = torch.stack(positives)
                negatives = torch.stack(negatives)
                Loss_contrust = Loss_contrust + loss_contrust(querys, positives, negatives)

            else:
                Loss_contrust = Loss_contrust

        Loss_contrust = Loss_contrust / B
        Loss = EPE(flow, flow_gt_src) + rate * Loss_contrust
        Loss.backward()
        optimizer.step()
        running_total_loss += Loss.item()
        running_total_loss_recover += Loss_contrust.item()
        train_writer.add_scalar("train_loss_per_iter", Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
            " Training: R_total_loss: %.3f/%.3f/%.3f/%.3f"
            % (running_total_loss / (i + 1), Loss.item(), running_total_loss_recover / (i + 1), Loss_contrust.item())
        )
    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net, val_loader, device, epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch["flow_src"].to(device)

            pred_flow, _, _, _ = net(
                mini_batch["src_img"].to(device),
                mini_batch["trg_img"].to(device),
                is_train=False
            )

            # Test Upsample PCK
            estimated_kps = flow2kps(
                mini_batch["src_kps"].to(device),
                pred_flow,
                mini_batch["n_pts"].to(device),
            )

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow, flow_gt)

            pck_array += eval_result["pck"]

            running_total_loss += Loss.item()
            pbar.set_description(
                " Validation R_total_loss: %.3f/%.3f"
                % (running_total_loss / (i + 1), Loss.item())
            )
        mean_pck = sum(pck_array) / len(pck_array)
        if mean_pck <= 1:
            raise RuntimeError
        print("####### Val mean pck: %.3f #######" % mean_pck)

    return running_total_loss / len(val_loader), mean_pck


def test_epoch(net, val_loader, device, epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch["flow_src"].to(device)
            # flow_gt_32 = F.interpolate(flow_gt, 32, None, "bilinear", False) / 2

            pred_flow, _, _, _ = net(
                mini_batch["src_img"].to(device),
                mini_batch["trg_img"].to(device),
                is_train=False
            )

            estimated_kps = flow2kps(
                mini_batch["src_kps"].to(device),
                pred_flow,
                mini_batch["n_pts"].to(device),
            )

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow, flow_gt)

            pck_array += eval_result["pck"]

            running_total_loss += Loss.item()
            pbar.set_description(
                " Test R_total_loss: %.3f/%.3f"
                % (running_total_loss / (i + 1), Loss.item())
            )
        mean_pck = sum(pck_array) / len(pck_array)
        print("####### Test mean pck: %.3f #######" % mean_pck)

    return running_total_loss / len(val_loader), mean_pck