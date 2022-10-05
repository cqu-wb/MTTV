#!/usr/bin/env python3


import argparse
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from mttv.data.helpers import get_data_loaders
from mttv.models.mttv import MTTV
from mttv.utils.logger import create_logger
from mttv.utils.utils import *

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args(parser):
    # the task/dataset, including fakeddit and weibo
    parser.add_argument("--task", type=str, default="fakeddit")

    # make sure : num_image_embeds=global_image_embeds+region_image_embeds
    parser.add_argument("--num_image_embeds", type=int, default=10)
    # super-parameter g -> the length of global visual embeddings
    parser.add_argument("--global_image_embeds", type=int, default=5)
    # super-parameter r -> the length of entity-level visual embeddings
    parser.add_argument("--region_image_embeds", type=int, default=5)

    # super-parameter N -> the number of transformer encoders
    parser.add_argument("--encoder_layer_num", type=int, default=6)

    # the dir of processed dataset
    parser.add_argument("--data_path", type=str, default="./data")
    # the dir of saving checkpoints and training history
    parser.add_argument("--savedir", type=str, default="./save")

    # for fakeddit, you can choice 2_way_label, 3_way_label or 6_way_label
    parser.add_argument("--label_type", type=str, default="label")

    # the pretrained bert model which is using to initial model
    # using bert-base-uncased for English, while using bert-base-chinese for chinese
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")

    # the real batch size equals batch_sz*gradient_accumulation_steps
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=20)

    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--model", type=str, default="mttv")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--load_checkpoint_path", type=str, default='')


def get_criterion(args):
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, args):
    total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0, },
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=args.warmup,
        t_total=total_steps,
    )
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in tqdm(data):
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())
            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)

    rep = classification_report(y_true=tgts, y_pred=preds, target_names=[str(label) for label in args.labels],
                                zero_division=False, digits=4)
    print(rep)
    rep_dict = classification_report(y_true=tgts, y_pred=preds, target_names=args.labels, zero_division=False,
                                     output_dict=True)
    metrics['rep_dict'] = rep_dict

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch):
    txt, segment, mask, img, tgt, regions = batch

    txt, img = txt.cuda(), img.cuda()
    mask, segment = mask.cuda(), segment.cuda()
    regions = regions.cuda()
    out = model(txt, mask, segment, img, regions)

    tgt = tgt.cuda()
    loss = criterion(out, tgt)
    return loss, out, tgt


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    logger = create_logger("%s/logfile.log" % args.savedir, args)

    train_loader, val_loader, test_loaders = get_data_loaders(args)
    criterion = get_criterion(args)

    model = MTTV(args)
    model.cuda()

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")

    train_history = []
    for i_epoch in range(start_epoch, args.max_epochs):
        logger.info('Epoch\t{}'.format(i_epoch + 1))
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)
        val_acc = metrics['acc']
        val_loss = metrics['loss']

        metrics = model_eval(i_epoch, test_loaders['test'], model, args, criterion)
        log_metrics("Test", metrics, args, logger)

        # 保存训练记录
        rep_dict = metrics['rep_dict']

        history_record = {
            'epoch': i_epoch + 1, 'train_loss': np.mean(train_losses),
            'val_loss': val_loss, 'val_acc': val_acc,
            'test_acc': metrics['acc']
        }
        for label in args.labels:
            for attr in ['precision', 'recall', 'f1-score', 'support']:
                history_record['{}_{}'.format(label, attr)] = rep_dict[label][attr]

        train_history.append(history_record)

        tuning_metric = (metrics["acc"])
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    # save training history
    df = pd.DataFrame(train_history)
    df.to_excel('save/{}/train_history.xlsx'.format(args.name), index=False)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    cli_main()
