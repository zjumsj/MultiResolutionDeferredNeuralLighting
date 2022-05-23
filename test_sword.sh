#### full dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/sword/baseline_sword_full --test_savedir ./val/baseline_sword_full/RotLight --test_dataset ./test/sword/Test_PL_RotY_exr
python train_baseline.py --test_only True --device 0 --checkpoint ./models/sword/baseline_sword_full --test_savedir ./val/baseline_sword_full/RotView --test_dataset ./test/sword/RotView_v1
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/sword/multires_sword_full --test_savedir ./val/multires_sword_full/RotLight --test_dataset ./test/sword/Test_PL_RotY_exr
python train_multires.py --test_only True --device 0 --checkpoint ./models/sword/multires_sword_full --test_savedir ./val/multires_sword_full/RotView --test_dataset ./test/sword/RotView_v1
#### 2000 photographs dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/sword/baseline_sword_s2000 --test_savedir ./val/baseline_sword_s2000/RotLight --test_dataset ./test/sword/Test_PL_RotY_exr
python train_baseline.py --test_only True --device 0 --checkpoint ./models/sword/baseline_sword_s2000 --test_savedir ./val/baseline_sword_s2000/RotView --test_dataset ./test/sword/RotView_v1
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/sword/multires_sword_s2000 --test_savedir ./val/multires_sword_s2000/RotLight --test_dataset ./test/sword/Test_PL_RotY_exr
python train_multires.py --test_only True --device 0 --checkpoint ./models/sword/multires_sword_s2000 --test_savedir ./val/multires_sword_s2000/RotView --test_dataset ./test/sword/RotView_v1

