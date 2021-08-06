rm -rf pp_data pp_csv pp_npy lr_csv lr_npy lr_pkl
mkdir pp_data pp_csv pp_npy lr_csv lr_npy lr_pkl
python data_preprocess.py
python feature_preprocess.py
python train_lr.py
