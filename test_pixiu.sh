#### full dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/pixiu/baseline_pixiu_full --test_savedir ./val/baseline_pixiu_full/RotLight --test_dataset ./test/pixiu/Rot_light_PL_exr --rescale_input 0.5
python train_baseline.py --test_only True --device 0 --checkpoint ./models/pixiu/baseline_pixiu_full --test_savedir ./val/baseline_pixiu_full/RotView --test_dataset ./test/pixiu/Rot_View_PL_exr --rescale_input 0.5
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/pixiu/multires_pixiu_full --test_savedir ./val/multires_pixiu_full/RotLight --test_dataset ./test/pixiu/Rot_light_PL_exr --rescale_input 0.5
python train_multires.py --test_only True --device 0 --checkpoint ./models/pixiu/multires_pixiu_full --test_savedir ./val/multires_pixiu_full/RotView --test_dataset ./test/pixiu/Rot_View_PL_exr --rescale_input 0.5
#### 2000 photographs dataset
# baseline model
python train_baseline.py --test_only True --device 0 --checkpoint ./models/pixiu/baseline_pixiu_s2000 --test_savedir ./val/baseline_pixiu_s2000/RotLight --test_dataset ./test/pixiu/Rot_light_PL_exr --rescale_input 0.5
python train_baseline.py --test_only True --device 0 --checkpoint ./models/pixiu/baseline_pixiu_s2000 --test_savedir ./val/baseline_pixiu_s2000/RotView --test_dataset ./test/pixiu/Rot_View_PL_exr --rescale_input 0.5
# our multi-res network
python train_multires.py --test_only True --device 0 --checkpoint ./models/pixiu/multires_pixiu_s2000 --test_savedir ./val/multires_pixiu_s2000/RotLight --test_dataset ./test/pixiu/Rot_light_PL_exr --rescale_input 0.5
python train_multires.py --test_only True --device 0 --checkpoint ./models/pixiu/multires_pixiu_s2000 --test_savedir ./val/multires_pixiu_s2000/RotView --test_dataset ./test/pixiu/Rot_View_PL_exr --rescale_input 0.5

