#### full dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/fish/baseline_fish_full --test_savedir ./val/baseline_fish_full/RotLight --test_dataset ./test/fish/RotLight_v2
python train_baseline.py --test_only True --device 0 --checkpoint ./models/fish/baseline_fish_full --test_savedir ./val/baseline_fish_full/RotView --test_dataset ./test/fish/RotView_v1
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/fish/multires_fish_full --test_savedir ./val/multires_fish_full/RotLight --test_dataset ./test/fish/RotLight_v2
python train_multires.py --test_only True --device 0 --checkpoint ./models/fish/multires_fish_full --test_savedir ./val/multires_fish_full/RotView --test_dataset ./test/fish/RotView_v1
#### 2000 photographs dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/fish/baseline_fish_s2000 --test_savedir ./val/baseline_fish_s2000/RotLight --test_dataset ./test/fish/RotLight_v2
python train_baseline.py --test_only True --device 0 --checkpoint ./models/fish/baseline_fish_s2000 --test_savedir ./val/baseline_fish_s2000/RotView --test_dataset ./test/fish/RotView_v1
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/fish/multires_fish_s2000 --test_savedir ./val/multires_fish_s2000/RotLight --test_dataset ./test/fish/RotLight_v2
python train_multires.py --test_only True --device 0 --checkpoint ./models/fish/multires_fish_s2000 --test_savedir ./val/multires_fish_s2000/RotView --test_dataset ./test/fish/RotView_v1

