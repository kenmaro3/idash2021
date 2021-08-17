rm -rf trained_model
mkdir /from_local/trained_model
python data_preprocess.py
python feature_preprocess.py
python train_lr.py
