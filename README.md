
# Public source code of MTTV (Multi-modal Transformer Using Two-level Visual Features for Fake News Detection)

## Recommended environment

The environment for our experiment:

```
python 3.8.8
pytorch 1.9.0+cu102
torchvision 0.2.1
pytorch-pretrained-bert 0.6.2
```

## Dataset

### (1) Download dataset

The Fakeddit and Weibo datasets which are preprocessed for our experiments are given in the directory `./data`, but do not include any image files.

You need to download the original datasets from the following links to obtain image files.

Fakeddit: [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)

Weibo: [https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)

### (2) Extract image features for the two datasets

```shell script
python extract_image_features.py --dataset_dir ./data/weibo/ --image_dir ${your_weibo_image_dir} --feature_dir ./data/weibo/
python extract_image_features.py --dataset_dir ./data/weibo/ --image_dir ${your_fakeddit_image_dir} --feature_dir ./data/fakeddit/
```

## Running

### (1) train MTTV on Fakediit with 6-way labels

```shell script
python train.py --task fakeddit --label_type 6_way_label --batch_sz 32 --gradient_accumulation_steps 20 --max_epochs 20 --name fakeddit_6_way --bert_model bert-base-uncased --global_image_embeds 5 --region_image_embeds 20 --num_image_embeds 25
```

### (2) train MTTV on Fakediit with 3-way labels

```shell script
python train.py --task fakeddit --label_type 3_way_label --batch_sz 32 --gradient_accumulation_steps 20 --max_epochs 20 --name fakeddit_3_way --bert_model bert-base-uncased --global_image_embeds 5 --region_image_embeds 20 --num_image_embeds 25
```

### (3) train MTTV on Fakediit with 2-way labels

```shell script
python train.py --task fakeddit --label_type 2_way_label --batch_sz 32 --gradient_accumulation_steps 20 --max_epochs 20 --name fakeddit_2_way --bert_model bert-base-uncased --global_image_embeds 5 --region_image_embeds 20 --num_image_embeds 25
```

### (4) train MTTV on Weibo

```shell script
python train.py --task weibo --label_type label --batch_sz 32 --gradient_accumulation_steps 1 --max_epochs 30 --seed 1 --name weibo --bert_model bert-base-chinese --global_image_embeds 5 --region_image_embeds 5 --num_image_embeds 10
```

### (5) use scalable classifier on Fakeddit with 6-way labels

Before using scalable classifier, you need to train MTTV on Fakediit with 6-way labels. Then, set `checkpoint_dir` and `checkpoint_name` correctly.

Using `parameter_tau` to set $ \tau $ of scalable classifier.

```shell script
python scalable_classifier.py --checkpoint_dir ./save/fakeddit_sim_test --checkpoint_name checkpoint_10.pt --parameter_tau 2.1
```

## Cite our work

If this code repository is helpful for your research, please cite our paper:

```
@article{wang2022multi,
  title={Multi-modal transformer using two-level visual features for fake news detection},
  author={Wang, Bin and Feng, Yong and Xiong, Xian-cai and Wang, Yong-heng and Qiang, Bao-hua},
  journal={Applied Intelligence},
  year={2022},
  publisher={Springer}
}
```
