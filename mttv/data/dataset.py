#!/usr/bin/env python3

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from mttv.utils.utils import numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.origin_data = [json.loads(l) for l in open(data_path, encoding='utf8')]
        self.data = self.origin_data
        self.label_type = args.label_type
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len - args.num_image_embeds
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.tokenizer(self.data[index]["text"])[: (self.args.max_seq_len - 1)]
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        segment = torch.ones(len(sentence))

        label = torch.LongTensor(
            [self.args.labels.index(self.data[index][self.args.label_type])]
        )

        # read image feature pre-extracted from Resnet-152
        feature_file = os.path.join(self.data_dir, 'visual_feature/{}.pkl'.format(self.data[index]["id"]))
        with open(feature_file, mode='rb') as f:
            image = pickle.load(f)['feature1']
            image = torch.Tensor(image)

        # read region feature (entity-level feature) pre-extracted from Faster-RCNN and Resnet-152
        img_name = self.data[index]["img"].replace('images/', '')
        feature_file = os.path.join(self.data_dir, 'region_feature/{}.pkl'.format(img_name))
        with open(feature_file, mode='rb') as f:
            region_image_feature = pickle.load(f)['features']
        region_feature_length = region_image_feature.shape[0]
        if region_feature_length < 20:
            region_image_feature = np.vstack([region_image_feature, np.zeros((20 - region_feature_length, 2048))])
        region_image_feature = torch.Tensor(region_image_feature)[:self.args.region_image_embeds, :]

        return sentence, segment, image, label, region_image_feature
