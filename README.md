# Based on TA's Sample Code Homework 1 ADL NTU 111 Fall

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
conda activate adl-hw1
pip install -r requirements.in
# otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```
## Slot tagging
```shell
python train_slot.py
```
