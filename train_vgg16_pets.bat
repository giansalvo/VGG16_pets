python main.py train -m vgg16 -w w_vgg16_c2.h5 -dr .\dataset_pets\ -c 2 -b 64
python main.py evaluate -m vgg16 -w w_vgg16_c2.h5 -dr .\dataset_pets\ -c 2
