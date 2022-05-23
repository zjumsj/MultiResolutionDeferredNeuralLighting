#### full dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/furscene/baseline_furscene_full --test_savedir ./val/baseline_furscene_full/RotLight --test_dataset ./test/furscene/RotLight_v1
python train_baseline.py --test_only True --device 0 --checkpoint ./models/furscene/baseline_furscene_full --test_savedir ./val/baseline_furscene_full/RotView --test_dataset ./test/furscene/RotView_v2
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/furscene/multires_furscene_full --test_savedir ./val/multires_furscene_full/RotLight --test_dataset ./test/furscene/RotLight_v1
python train_multires.py --test_only True --device 0 --checkpoint ./models/furscene/multires_furscene_full --test_savedir ./val/multires_furscene_full/RotView --test_dataset ./test/furscene/RotView_v2
#### 2000 photographs dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/furscene/baseline_furscene_s2000 --test_savedir ./val/baseline_furscene_s2000/RotLight --test_dataset ./test/furscene/RotLight_v1
python train_baseline.py --test_only True --device 0 --checkpoint ./models/furscene/baseline_furscene_s2000 --test_savedir ./val/baseline_furscene_s2000/RotView --test_dataset ./test/furscene/RotView_v2
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/furscene/multires_furscene_s2000 --test_savedir ./val/multires_furscene_s2000/RotLight --test_dataset ./test/furscene/RotLight_v1
python train_multires.py --test_only True --device 0 --checkpoint ./models/furscene/multires_furscene_s2000 --test_savedir ./val/multires_furscene_s2000/RotView --test_dataset ./test/furscene/RotView_v2

