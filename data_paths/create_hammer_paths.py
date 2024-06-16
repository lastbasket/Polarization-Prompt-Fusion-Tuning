import os

DATA_ROOT = "/root/autodl-tmp/yiming/datasets/polar_hammer"
PATH_LIST_TXT_TEMPLATE = "hammer_MODE.txt"
VAL_SCENES = ["scene11", "scene12"]
TEST_SCENES = ["scene13", "scene14"]
# meaning scenes 2-10 will be used as training sets

def is_target_folder(path):
    return ("scene" in path) and ("traj" in path) and ("naked" not in path) 

train_paths = []
val_paths = []
test_paths = []
cnt = 0
for d1 in os.listdir(DATA_ROOT):

    d1_full = os.path.join(DATA_ROOT, d1)

    if not os.path.isdir(d1_full) or not is_target_folder(d1):
        continue

    for d2 in os.listdir(d1_full):

        d2_full = os.path.join(d1_full, d2)

        if not os.path.isdir(d2_full) or not "rgb" in d2:
            continue
        
        for image in os.listdir(d2_full):
            full_path = os.path.join(d2_full, image)
            assigned = False

            for p in VAL_SCENES:
                if p in full_path:
                    val_paths.append(full_path)
                    assigned = True

            if not assigned:
                for p in TEST_SCENES:
                    if p in full_path:
                        test_paths.append(full_path)
                        assigned = True

            if not assigned:
                cnt += 1    
                # if cnt % 3 == 0: # drop one sample per 3 samples
                #     continue
                train_paths.append(full_path)

print("--> Collected train samples {}".format(len(train_paths)))
print("--> Collected val samples {}".format(len(val_paths)))
print("--> Collected test samples {}".format(len(test_paths)))
print("--> Collected in total {} samples".format(len(train_paths) + len(val_paths) + len(test_paths)))

path_lists = {"train": train_paths, "val": val_paths, "test": test_paths}
for mode in ["train", "val", "test"]:
    with open(PATH_LIST_TXT_TEMPLATE.replace("MODE", mode), "w") as file:
        for p in path_lists[mode]:
            p = p.replace(DATA_ROOT, "DATA_ROOT")
            file.write(p+"\n")