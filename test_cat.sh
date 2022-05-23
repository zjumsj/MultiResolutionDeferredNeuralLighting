#### full dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/cat/baseline_cat_full --test_savedir ./val/baseline_cat_full/RotLight --test_dataset ./test/cat/RotLight_v2
python train_baseline.py --test_only True --device 0 --checkpoint ./models/cat/baseline_cat_full --test_savedir ./val/baseline_cat_full/RotView --test_dataset ./test/cat/RotView_v2
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/cat/multires_cat_full --test_savedir ./val/multires_cat_full/RotLight --test_dataset ./test/cat/RotLight_v2
python train_multires.py --test_only True --device 0 --checkpoint ./models/cat/multires_cat_full --test_savedir ./val/multires_cat_full/RotView --test_dataset ./test/cat/RotView_v2
#### 1000 photographs dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/cat/baseline_cat_s1000 --test_savedir ./val/baseline_cat_s1000/RotLight --test_dataset ./test/cat/RotLight_v2
python train_baseline.py --test_only True --device 0 --checkpoint ./models/cat/baseline_cat_s1000 --test_savedir ./val/baseline_cat_s1000/RotView --test_dataset ./test/cat/RotView_v2
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/cat/multires_cat_s1000 --test_savedir ./val/multires_cat_s1000/RotLight --test_dataset ./test/cat/RotLight_v2
python train_multires.py --test_only True --device 0 --checkpoint ./models/cat/multires_cat_s1000 --test_savedir ./val/multires_cat_s1000/RotView --test_dataset ./test/cat/RotView_v2

