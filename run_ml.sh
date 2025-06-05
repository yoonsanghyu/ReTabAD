# for ML model
datasets=(breastw cardio cardiotocography cover hepatitis lymphography vertebral wbc wine yeast)
for dataset in "${datasets[@]}"; do
    python main.py --data_name ${dataset} --preprocess standard --cat_encoding int
    # python main.py --data_name ${dataset} --preprocess standard --cat_encoding onehot
    python main.py --data_name ${dataset} --preprocess minmax --cat_encoding int
    # python main.py --data_name ${dataset} --preprocess minmax --cat_encoding onehot
    python main.py --data_name ${dataset} --cat_encoding int
    # python main.py --data_name ${dataset} --cat_encoding onehot
done