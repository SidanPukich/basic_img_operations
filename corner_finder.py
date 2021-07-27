import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from concurrent import futures
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
    # print(f"{sub_img.shape} {corner.shape} {pos}")
    # print("in count sum")
    corner_w = corner.shape[1]
    corner_h = corner.shape[0]
    
    # if pos[0] % 100 == 0 and pos[1] % 100 == 0:
    #     print(f"count_sum {pos}")

    csum = 0

    # if np.sum(sub_img - 255) == 0:
    #     return (np.inf, pos)
    
    # for ich in range(corner_h):
    #     for icw in range(corner_w):
    #         csum += (int(sub_img[icw, ich]) - int(corner[icw, ich])) ** 2

    # csum = np.sum((sub_img - corner) ** 2)

    csum = np.sum(np.bitwise_xor(sub_img, corner))

    n = corner_h * corner_w
    csum /= n

    # if pos[0] % 100 == 0 and pos[1] % 100 == 0:
    #     print(f"count_sum {pos, csum, os.getpid()}")

    return (csum, pos)  

def img_iter(img, corner):
    
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    corner_w = corner.shape[1]
    corner_h = corner.shape[0]

    print(f"shape {img.shape}")
    for ih in range(img_h - corner_h + 1):
        # if ih % 100 == 0:
        #     print(ih)
        for iw in range(img_w - corner_w + 1):
            # if iw % 100 == 0 and ih % 100 == 0:
            #     print(f"img_iter {iw, ih}")
            # if img[ih:ih + corner_h, iw:iw + corner_w].shape != (52, 52):
            #     print(f"caution {iw, ih}")
            yield img[ih:ih + corner_h, iw:iw + corner_w], corner, (iw, ih)
    print("img iter prepared")        
        
    

def find_corner(img, corner):
#     img_w = img.shape[1]
#     img_h = img.shape[0]
    
#     corner_w = corner.shape[1]
#     corner_h = corner.shape[0]
    
    smallest_sum = math.inf
    corner_pos = (-1,-1)
    
    # with futures.ProcessPoolExecutor(max_workers=50) as ex:
    #     sums = list(ex.map(count_sum, img_iter(img, corner), timeout=None))
    with Pool() as pool:
        sums = list(pool.starmap(count_sum, img_iter(img, corner)))

    print(f"sums counted {len(sums)}")

    for csum, pos in sums:
        if csum < smallest_sum:
            smallest_sum = csum
            corner_pos = pos        
    return corner_pos, smallest_sum


print("hi")
print(f"cpu num {os.cpu_count()}")

path_evaluation = "data/boegen/Bogen1.jpg"
path_ul = "data/masks/ul.png"
path_ur = "data/masks/ur.png"
path_ll = "data/masks/ll.png"
path_lr = "data/masks/lr.png"

evaluation = cv2.imread(path_evaluation, cv2.IMREAD_GRAYSCALE)
ul = cv2.imread(path_ul, cv2.IMREAD_GRAYSCALE)
ur = cv2.imread(path_ur, cv2.IMREAD_GRAYSCALE)
ll = cv2.imread(path_ll, cv2.IMREAD_GRAYSCALE)
lr = cv2.imread(path_lr, cv2.IMREAD_GRAYSCALE)

if evaluation is not None and ul is not None and ur is not None and ll is not None and lr is not None:
    print("images Loaded")
else:
    print("images not found")


ul_o = find_ul_offset(ul)
ur_o = find_ur_offset(ur)
ll_o = find_ll_offset(ll)
lr_o = find_lr_offset(lr)

print(f"detected offset for ul: {ul_o}")
print(f"detected offset for ur: {ur_o}")
print(f"detected offset for ll: {ll_o}")
print(f"detected offset for lr: {lr_o}")

start = time.perf_counter()
ul_pos, csum_ul = find_corner(evaluation, ul)
endul = time.perf_counter()
ur_pos, csum_ur = find_corner(evaluation, ur)
endur = time.perf_counter()
ll_pos, csum_ll = find_corner(evaluation, ll)
endll = time.perf_counter()
lr_pos, csum_lr = find_corner(evaluation, lr)
endlr = time.perf_counter()

# print(f"detected pos for ul: {ul_pos, csum_ul}")
# print(f"detected pos for ll: {ll_pos, csum_ll}")
print(f"detected pos for ul: {ul_pos} in: {-start + endul} sec")
print(f"detected pos for ur: {ur_pos} in: {-endul + endur} sec")
print(f"detected pos for ll: {ll_pos} in: {-endur + endll} sec")
print(f"detected pos for lr: {lr_pos} in: {-endll + endlr} sec")
print(f"total time: {start - endlr} sec")




# corner_w = ul.shape[1]
# corner_h = ul.shape[0]
# img_ul = cv2.cvtColor(evaluation[ul_pos[1]:ul_pos[1] + corner_h, ul_pos[0]:ul_pos[0] + corner_w], cv2.COLOR_GRAY2BGR)
# img_ur = cv2.cvtColor(evaluation[ur_pos[1]:ll_pos[1] + corner_h, ll_pos[0]:ll_pos[0] + corner_w], cv2.COLOR_GRAY2BGR)
# img_ll = cv2.cvtColor(evaluation[ll_pos[1]:ll_pos[1] + corner_h, ll_pos[0]:ll_pos[0] + corner_w], cv2.COLOR_GRAY2BGR)
# img_lr = cv2.cvtColor(evaluation[ll_pos[1]:ll_pos[1] + corner_h, ll_pos[0]:ll_pos[0] + corner_w], cv2.COLOR_GRAY2BGR)

# cv2.imwrite("sub_img_ul.jpg", img_ul)
# cv2.imwrite("sub_img_ll.jpg", img_ll)

# evaluation_marked = cv2.cvtColor(evaluation_marked, cv2.COLOR_GRAY2RGB)
evaluation = cv2.cvtColor(evaluation, cv2.COLOR_GRAY2RGB)
ul = cv2.cvtColor(ul, cv2.COLOR_GRAY2RGB)
ur = cv2.cvtColor(ur, cv2.COLOR_GRAY2RGB)
ll = cv2.cvtColor(ll, cv2.COLOR_GRAY2RGB)
lr = cv2.cvtColor(lr, cv2.COLOR_GRAY2RGB)


evaluation_marked = evaluation

ul_pos = (ul_pos[0] + ul_o[0], ul_pos[1] + ul_o[1])
ur_pos = (ur_pos[0] + ur_o[0], ur_pos[1] + ur_o[1])
ll_pos = (ll_pos[0] + ll_o[0], ll_pos[1] + ll_o[1])
lr_pos = (lr_pos[0] + lr_o[0], lr_pos[1] + lr_o[1])


evaluation_marked = cv2.circle(evaluation_marked, ul_pos, radius=15, color=(0, 0, 255), thickness=3)
evaluation_marked = cv2.circle(evaluation_marked, ur_pos, radius=15, color=(0, 255, 0), thickness=3)
evaluation_marked = cv2.circle(evaluation_marked, ll_pos, radius=15, color=(255, 0, 0), thickness=3)
evaluation_marked = cv2.circle(evaluation_marked, lr_pos, radius=15, color=(0, 0, 0), thickness=3)

cv2.imwrite("test.jpg", evaluation_marked)
# TODO: используем ROI если результат поиска плохой, проверяем весь лист


if evaluation is not None and ul is not None and ur is not None and ll is not None and lr is not None:
    f, axarr = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
    axarr[0, 0].imshow(evaluation)
    axarr[0, 1].imshow(evaluation_marked)
    axarr[1, 0].imshow(ul)
    axarr[1, 1].imshow(ur)
    axarr[2, 0].imshow(ll)
    axarr[2, 1].imshow(lr)
    axarr[0, 0].set_title("Evaluation")
    axarr[0, 1].set_title("Evaluation Marked")
    axarr[1, 0].set_title("UL")
    axarr[1, 1].set_title("UR")
    axarr[2, 0].set_title("LL")
    axarr[2, 1].set_title("LR")
#     axarr[0, 0].axis("off")
#     axarr[0, 1].axis("off")
#     axarr[1, 0].axis("off")
#     axarr[1, 1].axis("off")
#     axarr[2, 0].axis("off")
#     axarr[2, 1].axis("off")
    
    
    plt.show()