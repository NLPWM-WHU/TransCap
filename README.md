# TransCap
Code and dataset of our paper "[Transfer Capsule Network for Aspect Level Sentiment Classification](https://www.aclweb.org/anthology/P19-1052)" accepted by ACL 2019.

## 1. Requirements
* python 3.6
* tensorflow 1.3.0
* spacy 1.9.0
* numpy 1.16.4
* scikit-learn 0.21.2

## 2. Usage
 We incorporate the training and evaluation of TransCap in the ```main.py```. Just run it as below.

```
CUDA_VISIBLE_DEVICES=0 python main.py --ASC restaurant --DSC yelp
```

## 3. Embeddings
 We have generated the word-idx mapping file and the word embedding file in ```./data/restaurant``` and ```./data/laptop```. If you want to generate them from scratch, follow the steps below. We take restaurant(ASC) + yelp(DSC) for an example.

* Download [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) and put it in ```./data```.
* Execute ```CUDA_VISIBLE_DEVICES=0 python main.py --ASC restaurant --DSC yelp --reuse_embedding False```.
* Related files will be generated in ```./data/restaurant```.

## 4. Run TransCap on Other Datasets
If you want to run TransCap on a new-coming dataset (e.g., 'XXX'), follow the instructions below.

* Create the folder ```./data/XXX``` , generate the ASC files, and put them in corresponding folders like ```./data/XXX/train```.
* Generate the DSC files (e.g., files start with 'YYY') and put them in ```./data/XXX/train```.
* Copy ```./data/restaurant/balance.py``` and put it in ```./data/XXX```.
* Run ```./data/XXX/balance.py``` to get balanced ASC files.
* Execute ```CUDA_VISIBLE_DEVICES=0 python main.py --ASC XXX --DSC YYY --reuse_embedding False``` to run TransCap on the XXX dataset.

## 5. Citation
If you find our code and dataset useful, please cite our paper.  
  
```
@inproceedings{chen2019transcap,
  author    = {Zhuang Chen and Tieyun Qian},
  title     = {Transfer Capsule Network for Aspect Level Sentiment Classification},
  booktitle = {ACL},
  pages     = {547--556},
  year      = {2019},
}
```

