python main.py train -m resnet50 -w w_resnet50 _c2.h5 -dr .\dataset_pets\ -c 2 -b 64
python main.py evaluate -m resnet50 -w w_resnet50 _c2.h5 -dr .\dataset_pets\ -c 2
