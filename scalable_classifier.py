import argparse
import os
import torch
from mttv.data.helpers import get_data_loaders
from mttv.models.mttv import MTTV, MTTV_WithScalableClassifier
from train import model_eval, get_criterion


def get_args(parser):
    # the source image dir
    parser.add_argument("--checkpoint_dir", type=str, default="./save/fakeddit_6_way")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_10.pt")
    parser.add_argument("--parameter_tau", type=float, default=1.0)


def test_scale_classifier(args):
    # load model args
    model_args_path = os.path.join(args.checkpoint_dir, "args.pt")
    model_args = torch.load(model_args_path)
    print("load model args from : ", model_args_path)

    # load MTTV model
    model_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    print("load trained model from : ", model_path)
    model = MTTV(model_args)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.cuda()
    model.eval()

    # get data_loaders
    train_loader, val_loader, test_loaders = get_data_loaders(model_args)
    test_loader = test_loaders['test']
    print("load data loader completed.")

    # test MTTV on Validation set and compute scaling_factors
    criterion = get_criterion(model_args)
    print("running model evaluate on Validation set ...")
    metrics = model_eval(99, val_loader, model, model_args, criterion)
    recalls = []
    for label in model_args.labels:
        recalls.append(metrics['rep_dict'][label]['recall'])
    print("the accuracy for each class : ", recalls)
    print("parameter tau = ", args.parameter_tau)
    scaling_factors = [(1 / i) ** args.parameter_tau for i in recalls]
    print("the scaling_factors are : ", scaling_factors)

    # construct mttv_with_scalable_classifier
    model = MTTV_WithScalableClassifier(model, scaling_factors)
    model.cuda()
    model.eval()
    print("construct MTTV_with_scalable_classifier completed.")

    # test mttv_with_scalable_classifier on test set
    print("running model evaluate on Test set ...")
    model_eval(99, test_loader, model, model_args, criterion)


def cli_main():
    parser = argparse.ArgumentParser(description="evaluate MTTV with scalabel classifier")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    test_scale_classifier(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    cli_main()
