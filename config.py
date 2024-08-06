# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = '/homesda/home/ybjia/osr_closed_set_all_you_need/'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = '/homesda/home/ybjia/osr_closed_set_all_you_need/open_set_recognition'        # directory to store experiment output (checkpoints, logs, etc)
save_dir = '/homesda/home/ybjia/osr_closed_set_all_you_need/open_set_recognition/methods/baseline/ensemble_entropy_test'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = '/homesda/home/ybjia/osr_closed_set_all_you_need/open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = '/homesda/home/ybjia/osr_closed_set_all_you_need/open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'


project_path = 'path/to/project'
# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root = f'{project_path}/datasets/cifar-10-python'                                    # CIFAR10
cifar_100_root = f'{project_path}/datasets/cifar-100'                                         # CIFAR100
cub_root = f'{project_path}/datasets/CUB'                                                     # CUB
aircraft_root = f'{project_path}/datasets/FGVC_Aircraft/fgvc-aircraft-2013b'                  # FGVC-Aircraft
mnist_root = f'{project_path}/datasets/mnist/'                                                # MNIST
pku_air_root = f'{project_path}/datasets/pku-air-300/AIR'                                     # PKU-AIRCRAFT-300
car_root = "/stanford_cars/cars_{}/"                                                          # Stanford Cars
meta_default_path = "/Cars/cars_{}.mat"
svhn_root = f'{project_path}/datasets/svhn'                                                   # SVHN
tin_train_root_dir = '/homesda/home/ybjia/Projects/datasets/tiny-imagenet-200/train'
tin_val_root_dir = '/homesda/home/ybjia/Projects/datasets/tiny-imagenet-200/val/images'
# tin_train_root_dir = f'{project_path}/datasets/tiny-imagenet-200/train'                       # TinyImageNet Train
# tin_val_root_dir = f'{project_path}/datasets/tiny-imagenet-200/val/images'                    # TinyImageNet Val
imagenet_root = f'{project_path}/datasets/imagenet'

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = f'{project_path}/data/open_set_splits'