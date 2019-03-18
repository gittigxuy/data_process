# Written by Hongkai Zhang

import numpy as np
import argparse
import json
from tqdm import tqdm


def box_overlap(box, gt, phase='iou'):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) NDArray
    gt : (K, 4) NDArray
    return: (N, K) NDArray, stores Max(0, intersection/union) or Max(0, intersection/area_box)
    """
    N = box.shape[0]
    K = gt.shape[0]
    target_shape = (N, K, 4)
    b_box = np.broadcast_to(np.expand_dims(box, axis=1), target_shape)
    b_gt = np.broadcast_to(np.expand_dims(gt, axis=0), target_shape)

    iw = (np.minimum(b_box[:, :, 2], b_gt[:, :, 2]) -
          np.maximum(b_box[:, :, 0], b_gt[:, :, 0]))
    ih = (np.minimum(b_box[:, :, 3], b_gt[:, :, 3]) -
          np.maximum(b_box[:, :, 1], b_gt[:, :, 1]))
    inter = np.maximum(iw, 0) * np.maximum(ih, 0)

    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_target_shape = (N, K)
    b_area_box = np.broadcast_to(np.expand_dims(area_box, axis=1), area_target_shape)
    b_area_gt = np.broadcast_to(np.expand_dims(area_gt, axis=0), area_target_shape)

    assert phase == 'iou' or phase == 'ioa'
    union = b_area_box + b_area_gt - inter if phase == 'iou' else b_area_box

    overlaps = np.maximum(inter / np.maximum(union, 1), 0)
    return overlaps


def load_file(path, phase='GT'):
    # each line of file is in JSON format
    assert phase in ['GT', 'DET']
    print('Loading {} from {} ...'.format(phase, path))

    with open(path, 'r') as f:
        lines = f.readlines()
    res_dict = dict()
    for line in lines:
        record = json.loads(line.strip())
        image_id = record['ID']
        boxes = record['gtboxes'] if phase == 'GT' else record['dtboxes']

        res = []
        for box in boxes:
            box['box'][2] += box['box'][0]
            box['box'][3] += box['box'][1]
            if phase == 'GT':
                if 'ignore' in box['extra'] and box['extra']['ignore'] == 1:
                    box['box'].append(1)
                else:
                    box['box'].append(0)
            else:
                box['box'].append(box['score'])
            res.append(box['box'])
        res = np.array(res)
        res_dict[image_id] = res

    print('Done.')
    return res_dict


def crowd_human_eval(gt_path, dt_path, thres=0.5):
    gts = load_file(gt_path, 'GT')
    dts = load_file(dt_path, 'DET')
    ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]

    all_dt_matched = list()
    all_score = list()
    num_gt = 0
    num_image = len(dts)
    for image_id, dtboxes in tqdm(dts.items()):
        assert image_id in gts.keys()
        gtboxes = gts[image_id]
        num_gt += len(np.where(gtboxes[:, 4] == 0)[0])
        gtboxes = gtboxes[np.argsort(gtboxes[:, 4]), :]
        dtboxes = dtboxes[np.argsort(dtboxes[:, 4])[::-1], :]
        gt_matched = np.zeros(gtboxes.shape[0])
        dt_matched = np.zeros(dtboxes.shape[0])

        overlaps_iou = box_overlap(dtboxes[:, :4], gtboxes[:, :4], 'iou')
        overlaps_ioa = box_overlap(dtboxes[:, :4], gtboxes[:, :4], 'ioa')
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[4] == 0:
                    overlap = overlaps_iou[i, j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlaps_ioa[i, j]
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos, 4] == 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0

        valid_inds = np.where(dt_matched != -1)[0]
        all_score.extend(dtboxes[valid_inds, 4])
        all_dt_matched.extend(dt_matched[valid_inds])
    sort_inds = np.argsort(all_score)[::-1]
    all_dt_matched = np.array(all_dt_matched)[sort_inds]

    # calculate mMR
    tp, fp = 0.0, 0.0
    fppiX, fppiY = list(), list()

    for dt_match in all_dt_matched:
        if dt_match == 1:
            tp += 1
        else:
            fp += 1

        recall = tp / num_gt
        missrate = 1.0 - recall
        fppi = fp / num_image
        fppiX.append(fppi)
        fppiY.append(missrate)
    fppiX = np.array(fppiX)

    score = list()
    for pos in ref:
        idx = np.where(fppiX >= pos)[0]
        argmin = len(fppiX) - 1 if len(idx) == 0 else idx[0]
        if argmin >= 0:
            score.append(fppiY[argmin])

    score = np.array(score)
    MR = np.exp(np.log(score).mean())
    return MR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', dest='dt_path', required=True)
    parser.add_argument('--gt', dest='gt_path', required=True)

    args = parser.parse_args()
    mMR = crowd_human_eval(args.gt_path, args.dt_path)
    print('mMR: {}'.format(mMR))
