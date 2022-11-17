python main.py train -m mobilenetv2 -w w_mobilenetv2_c2.h5 -dr .\dataset_pets\ -c 2 -b 128
python main.py evaluate -m mobilenetv2 -w w_mobilenetv2_c2.h5 -dr .\dataset_pets\ -c 2 
