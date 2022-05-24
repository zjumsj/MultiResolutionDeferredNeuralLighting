device_id=0
echo $device_id
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 0 --max_steps 20000 --stage 0 --checkpoint None
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 20000 --max_steps 20000 --stage 1
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 40000 --max_steps 20000 --stage 2
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 60000 --max_steps 20000 --stage 3
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 80000 --max_steps 20000 --stage 4
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 100000 --max_steps 20000 --stage 5
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 120000 --max_steps 20000 --stage 6
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 140000 --max_steps 20000 --stage 7
python train_multires.py --config config/config_progressive.txt --device $device_id --start_step 160000 --max_steps 100000 --stage 8


