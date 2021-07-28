import argparse
import cv2
import time
import numpy as np
import math
from multiprocessing import Pool
import os

def find_ul_offset(ul):
    o_y = -1
    o_x = -1

    for y in range(ul.shape[0]):
        if ul[y, ul.shape[1] - 1] == 0:
            o_y = y
            break

    for x in range(ul.shape[1]):
        if ul[ul.shape[0] - 1, x] == 0:
            o_x = x
            break

    return (o_x, o_y) 
        
def find_ur_offset(ur):
    o_y = -1
    o_x = -1

    for y in range(ur.shape[0]):
        if ur[y, 0] == 0:
            o_y = y
            break

    for x in range(ur.shape[1]):
        if ur[ur.shape[0] - 1, ur.shape[1] - x - 1] == 0:
            o_x = ur.shape[1] - x - 1
            break

    return (o_x, o_y)

def find_ll_offset(ll):
    o_y = -1
    o_x = -1

    for y in range(ll.shape[0]):
        if ll[ll.shape[0] - y - 1, ll.shape[1] - 1] == 0:
            o_y = ll.shape[0] - y - 1
            break

    for x in range(ll.shape[1]):
        if ll[0, x] == 0:
            o_x = x
            break

    return (o_x, o_y)

def find_lr_offset(lr):
    o_y = -1
    o_x = -1

    for y in range(lr.shape[0]):
        if lr[lr.shape[0] - y - 1, 0] == 0:
            o_y = lr.shape[0] - y - 1
            break

    for x in range(lr.shape[1]):
        if lr[0, lr.shape[1] - x - 1] == 0:
            o_x = lr.shape[1] - x - 1
            break

    return (o_x, o_y)

def count_sum(sub_img, corner, pos):
    corner_w = corner.shape[1]
    corner_h = corner.shape[0]
    csum = 0
    csum = np.sum(np.bitwise_xor(sub_img, corner))
    n = corner_h * corner_w
    csum /= n

    return (csum, pos)  

def img_iter(img, corner, roi):
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    corner_w = corner.shape[1]
    corner_h = corner.shape[0]

    roi_offset_x = (int)(roi[0] * img_w)
    roi_offset_y = (int)(roi[1] * img_h)
    roi_width = (int)(roi[2] * img_w)
    roi_height = (int)(roi[3] * img_h)

    # print(f"{roi_offset_x, img_w - corner_w + 1, roi_width, roi_offset_y, img_h - corner_h + 1, roi_height}")
    # print(f"shape {img.shape}")
    for ih in range(roi_offset_y, roi_offset_y + roi_height - corner_h + 1):
        for iw in range(roi_offset_x, roi_offset_x + roi_width - corner_w + 1):
            # print(f"{ih, iw, corner_h, corner_w, img_h - corner_h + 1 - roi_height, img_w - corner_w + 1 - roi_width}")
            yield img[ih:ih + corner_h, iw:iw + corner_w], corner, (iw, ih)
    # print("img iter prepared")            

def find_corner(img, corner, roi):
    smallest_sum = math.inf
    corner_pos = (-1,-1)

    with Pool() as pool:
        sums = list(pool.starmap(count_sum, img_iter(img, corner, roi)))

    # print(f"sums counted {len(sums)}")

    for csum, pos in sums:
        if csum < smallest_sum:
            smallest_sum = csum
            corner_pos = pos        
    return corner_pos, smallest_sum

def print_roi(img, roi, path):
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    roi_offset_x = (int)(roi[0] * img_w)
    roi_offset_y = (int)(roi[1] * img_h)
    roi_width = (int)(roi[2] * img_w)
    roi_height = (int)(roi[3] * img_h)

    roi_img = img[roi_offset_y:roi_offset_y + roi_height, roi_offset_x: roi_offset_x + roi_width]
    
    print(f"{path} has shape {roi_img.shape}")

    cv2.imwrite(path, roi_img)

def find_corners(boegen_path, masks_path, rois):
    if not os.path.exists(boegen_path):
        raise Exception(f"Boegen folder {boegen_path} doesn't exist")

    boegen = []

    for f in os.listdir(boegen_path):
        if os.path.isdir(f):
            continue
        boegen += [os.path.join(boegen_path, f)]

    masks = [
        os.path.join(masks_path, "ul.png"),
        os.path.join(masks_path, "ur.png"),
        os.path.join(masks_path, "ll.png"),
        os.path.join(masks_path, "lr.png")
        ]

    for mask in masks:
        if not os.path.exists(mask):
            raise Exception(f"File {mask} doesn't exist")

    ul = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
    ur = cv2.imread(masks[1], cv2.IMREAD_GRAYSCALE)
    ll = cv2.imread(masks[2], cv2.IMREAD_GRAYSCALE)
    lr = cv2.imread(masks[3], cv2.IMREAD_GRAYSCALE)

    ul_o = find_ul_offset(ul)
    ur_o = find_ur_offset(ur)
    ll_o = find_ll_offset(ll)
    lr_o = find_lr_offset(lr)

    corners = {}

    test_dir = os.path.join(os.path.dirname(boegen_path), "test")
    
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, f))
        os.rmdir(test_dir)
    os.mkdir(test_dir)

    i = 0
    for path_evaluation in boegen:
        evaluation = cv2.imread(path_evaluation, cv2.IMREAD_GRAYSCALE)

        # start = time.perf_counter()
        ul_pos, csum_ul = find_corner(evaluation, ul, rois[0])
        # endul = time.perf_counter()
        ur_pos, csum_ur = find_corner(evaluation, ur, rois[1])
        # endur = time.perf_counter()
        ll_pos, csum_ll = find_corner(evaluation, ll, rois[2])
        # endll = time.perf_counter()
        lr_pos, csum_lr = find_corner(evaluation, lr, rois[3])
        # endlr = time.perf_counter()

        # print(f"detected pos for ul: {ul_pos} in: {-start + endul} sec")
        # print(f"detected pos for ur: {ur_pos} in: {-endul + endur} sec")
        # print(f"detected pos for ll: {ll_pos} in: {-endur + endll} sec")
        # print(f"detected pos for lr: {lr_pos} in: {-endll + endlr} sec")
        # print(f"total time: {-start + endlr} sec")

        evaluation = cv2.cvtColor(evaluation, cv2.COLOR_GRAY2RGB)

        ul_pos = (ul_pos[0] + ul_o[0], ul_pos[1] + ul_o[1])
        ur_pos = (ur_pos[0] + ur_o[0], ur_pos[1] + ur_o[1])
        ll_pos = (ll_pos[0] + ll_o[0], ll_pos[1] + ll_o[1])
        lr_pos = (lr_pos[0] + lr_o[0], lr_pos[1] + lr_o[1])

        evaluation_marked = cv2.circle(evaluation, ul_pos, radius=15, color=(0, 0, 255), thickness=3)
        evaluation_marked = cv2.circle(evaluation, ur_pos, radius=15, color=(0, 255, 0), thickness=3)
        evaluation_marked = cv2.circle(evaluation, ll_pos, radius=15, color=(255, 0, 0), thickness=3)
        evaluation_marked = cv2.circle(evaluation, lr_pos, radius=15, color=(0, 0, 0), thickness=3)

        cv2.imwrite(os.path.join(test_dir, path_evaluation), evaluation_marked)
        corners[path_evaluation] = (ul_pos, ur_pos, ll_pos, lr_pos)
        i += 1
        print(f"[{i} / {len(boegen)}]: Corners detected on {path_evaluation}. Corners are {(ul_pos, ur_pos, ll_pos, lr_pos)}")
    return corners

def draw_corners_n_rois(corners, vis_path, rois):
    for i in corners.items():
        f = i[0]
        cs = i[1]

        evaluation = cv2.imread(f)

        for corner in cs:
            evaluation = cv2.circle(evaluation, corner, radius=15, color=(0, 0, 255), thickness=3)

        img_w = evaluation.shape[1]
        img_h = evaluation.shape[0]
        
        for roi in rois:
            roi_offset_x = (int)(roi[0] * img_w)
            roi_offset_y = (int)(roi[1] * img_h)
            roi_width = (int)(roi[2] * img_w)   
            roi_height = (int)(roi[3] * img_h)  

            evaluation = cv2.rectangle(evaluation, (roi_offset_x, roi_offset_y), (roi_offset_x + roi_width, roi_offset_y + roi_height), color=(255, 0, 0), thickness=3)

        cv2.imwrite(os.path.join(vis_path, os.path.basename(f)), evaluation)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--boegen_path", default="data/boegen", type=str, help="Path to evaluations folder")
    parser.add_argument("-m", "--masks_path", default="data/masks", type=str, help="Path to masks folder")
    parser.add_argument("-v", "--vis_path", type=str, help="Debug corners detection. Path to folder")
    opt = parser.parse_args()

    # lux, luy, w, h 
    ul_roi = (0.3, 0.1, 0.2, 0.2)
    ur_roi = (0.75, 0.1, 0.25, 0.2)
    # ll_roi = (0, 0.85, 0.2, 0.15)
    ll_roi = (0.3, 0.85, 0.2, 0.15)
    lr_roi = (0.85, 0.8, 0.15, 0.2)

    rois = (ul_roi, ur_roi, ll_roi, lr_roi)

    try:
        print("Starting corners detection...")
        start_time = time.perf_counter()
        corners = find_corners(opt.boegen_path, opt.masks_path, rois)
        finish_time = time.perf_counter()
        print(f"Corners detection is finished in {finish_time - start_time} sec for {len(corners)} Boegen")

        if opt.vis_path:
            draw_corners_n_rois(corners, opt.vis_path, rois)
    except Exception as e:
        print(e)    



