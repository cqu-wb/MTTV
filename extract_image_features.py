import argparse
import json
import os
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args(parser):
    parser.add_argument("--dataset_dir", type=str, default="./data/fakeddit/")
    parser.add_argument("--image_dir", type=str, default="./data/fakeddit/")
    parser.add_argument("--feature_dir", type=str, default="./data/fakeddit/")


def extract_global_visual_feature(args):
    resnet = torchvision.models.resnet152(pretrained=True)

    resnet_pool = list(resnet.children())[-2]
    resnet_pool.eval()

    modules = list(resnet.children())[0:-2]
    resnet = nn.Sequential(*modules).cuda()
    resnet.eval()

    resnet_image_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    error_records = []
    if not os.path.exists(os.path.join(args.feature_dir, 'visual_feature')):
        os.makedirs(os.path.join(args.feature_dir, 'visual_feature'))

    def deal_batch(records):
        try:
            with torch.no_grad():
                img_tensors = []
                for record in records:
                    img_path = os.path.join(args.image_dir, record['img'])
                    img = Image.open(img_path).convert('RGB')
                    img = resnet_image_transforms(img).cuda()
                    img_tensors.append(img)
                imgs = torch.stack(img_tensors)
                feature1s = resnet(imgs).cpu()  # feature size 2048*7*7
                feature2s = resnet_pool(feature1s).squeeze(-1).squeeze(-1).cpu()  # feature size 2048
                # save feature
                for i, record in enumerate(records):
                    img_id = record['id']
                    feature1 = feature1s[i].numpy()
                    feature2 = feature2s[i].numpy()
                    save_feature_path = os.path.join(args.feature_dir, 'visual_feature/{}.pkl'.format(img_id))
                    with open(save_feature_path, mode='wb') as plk_f:
                        pickle.dump({'feature1': feature1, 'feature2': feature2}, plk_f)
        except Exception as e:
            print(e)
            error_records.append(records)

    batch_size = 150
    for f in ['test.jsonl', 'dev.jsonl', 'train.jsonl']:
        f = os.path.join(args.dataset_dir, f)
        with open(f, encoding='utf8') as data:
            records = []
            for line in tqdm(data):
                line = json.loads(line)
                records.append(line)
                if len(records) >= batch_size:
                    deal_batch(records)
                    records.clear()
            if len(records) > 0:
                deal_batch(records)


def extract_entity_level_visual_feature(args):
    max_boxes = 20
    feature_size = 2048
    device = torch.device('cuda')
    # Faster-RCNN
    fast_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    fast_rcnn.eval()
    # Resnet
    resnet = torchvision.models.resnet152(pretrained=True)
    modules = list(resnet.children())[0:-1]
    resnet = nn.Sequential(*modules).to(device)
    resnet.eval()
    # transforms
    resnet_image_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    toTensor = transforms.ToTensor()

    if not os.path.exists(os.path.join(args.feature_dir, 'region_feature')):
        os.makedirs(os.path.join(args.feature_dir, 'region_feature'))

    def get_region_and_feature(img_path, feature_path):
        with torch.no_grad():
            image = Image.open(img_path).convert("RGB")
            img = toTensor(image)
            img = img.unsqueeze(0).to(device)
            output = fast_rcnn(img)[0]
            box_num = min(max_boxes, output['boxes'].shape[0])
            regions = torch.zeros((box_num, 3, 224, 224))

            for i in range(box_num):
                box = output['boxes'][i].int().tolist()
                region = image.crop(box).convert("RGB")
                region = resnet_image_transforms(region)
                regions[i] = region

            if box_num == 0:
                features = torch.zeros([0, 2048])
            else:
                regions = regions.to(device)
                features = resnet(regions).squeeze(-1).squeeze(-1)
            save_dict = {'box_num': box_num, 'features': features.cpu().numpy(),
                         'boxes': output['boxes'][0:box_num].int().cpu().numpy(),
                         'labels': output['labels'][0:box_num].cpu().numpy(),
                         'scores': output['scores'][0:box_num].cpu().numpy()}
            with open(feature_path, mode='wb') as f:
                pickle.dump(save_dict, f)

    image_list = []
    error_image_list = []
    for f in ['train.jsonl', 'test.jsonl', 'dev.jsonl']:
        with open(os.path.join(args.dataset_dir, f), encoding='utf8') as ff:
            images = [json.loads(line)['img'] for line in ff]
            image_list += images
    for image_name in tqdm(image_list):
        try:
            image_path = os.path.join(args.image_dir, image_name)
            feature_path = os.path.join(args.feature_dir, "region_feature/" + image_name.replace("images/", '') + '.pkl')
            get_region_and_feature(image_path, feature_path)
        except Exception as e:
            error_image_list.append(image_name)
            print('error:' + image_name)
            print(e)


def cli_main():
    parser = argparse.ArgumentParser(description="extract image features")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    print('begin extract global_visual_feature...')
    extract_global_visual_feature(args)
    print('begin extract entity_level__visual_feature...')
    extract_entity_level_visual_feature(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
