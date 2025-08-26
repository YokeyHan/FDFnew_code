import argparse
import torch
import json
from pathlib import Path

def organize_file_structure(_args: argparse.Namespace):
    save_dir = Path(_args.save_dir)  
    save_dir = save_dir / _args.dataset  
    save_dir = save_dir / _args.model_name 
    save_dir = save_dir / str(_args.train_times) 
    train_settings = "i{}_o{}".format(
        _args.input_len,
        _args.pred_len
    )

    exp_save_dir = save_dir / train_settings
    exp_save_dir.mkdir(
        parents=True, exist_ok=True
    )
    args_save_path = exp_save_dir / "args.json"
    with open(args_save_path, "w") as f:
        json.dump(vars(_args), f, indent=4)
    scores_save_path = exp_save_dir / "scores.txt"
    _args.scores_save_path = scores_save_path

    log_dir = save_dir / "logs"
    _args.log_dir = log_dir
    log_dir.mkdir(
        parents=True, exist_ok=True
    )

    _args.train_settings = train_settings
    train_save_dir = exp_save_dir / "train"
    _args.train_save_dir = train_save_dir

    if _args.task_name == "train":
        train_save_dir.mkdir(parents=True, exist_ok=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDF Args")

    # basic config
    parser.add_argument("--data_dir", type=str, default="datasets", help="data directory")
    parser.add_argument("--dataset", type=str, default="METR-LA", help="dataset name") #ETTh1, METR-LA
    parser.add_argument("--save_dir", type=str, default="results_test", help="save results or train models directory")

    # data loader
    parser.add_argument("--train_batch_size", type=int, default=4, help="batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=4, help="batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=4, help="batch size for testing")
    parser.add_argument("--scale", action="store_true", help="scale data", default=True)
    parser.add_argument("--input_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="predicted sequence length")
    parser.add_argument("--feature_dim", type=int, default=207*4, help="number of features") #207  325

    #model define
    parser.add_argument("--model_name", type=str, default="FDF", help="model name")
    parser.add_argument("--time_steps", type=int, default=50, help="time steps in diffusion")
    parser.add_argument("--scheduler", type=str, default="cosine", help="scheduler in diffusion")
    parser.add_argument("--MLP_hidden_dim", type=int, default=512, help="MLP hidden dim")
    parser.add_argument("--emb_dim", type=int, default=4, help="emb dim")  #207 325
    parser.add_argument("--patch_size", type=int, default=4, help="patch size")


    # cluster (新增两项)
    parser.add_argument("--z_dim", type=int, default=4, help="subspace dimension")
    parser.add_argument("--num_clusters", type=int, default=4, help="number of clusters")
    parser.add_argument("--ST_channels", type=int, default=4, help="channels in ST encoder")
    parser.add_argument("--batch_size", type=int, default=4, help="training batch size for clustering")
    #parser.add_argument("--cluster_tau", type=float, default=0.5, help="temperature τ for Sinkhorn clustering")

    #optimization
    parser.add_argument("--train_flag", type=int, help="training or not", default=1)
    parser.add_argument("--train_times", type=int, default=1, help="times of training")
    parser.add_argument("--task_name", type=str, default="train", help="task name: train")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs for train")
    parser.add_argument("--eval_frequency", type=int, default=1, help="evaluation frequency for train")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="train learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="train weight decay")
    parser.add_argument("--lr_decay", type=float, default=0.99, help="train learning rate decay")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss type")


    #gpu
    parser.add_argument("--verbose", action="store_true", help="verbose", default=True)
    parser.add_argument("--use_tqdm", action="store_true", help="use tqdm", default=True)
    parser.add_argument("--seed", type=int, default=2024, help="fixed random seed")
    parser.add_argument("--device", type=str, default="cuda:2", help="device")
    
    _args = parser.parse_args()
    organize_file_structure(_args)

    _args.device = torch.device(_args.device)
    return _args
