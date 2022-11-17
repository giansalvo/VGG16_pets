python main.py train -m densenet201 -w w_densenet201_c2.h5 -dr .\dataset_pets\ -c 2 -b 32 
python main.py evaluate -m densenet201 -w w_densenet201_c2.h5 -dr .\dataset_pets\ -c 2 
