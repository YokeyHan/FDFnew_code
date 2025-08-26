from argparse import Namespace
from .dataset import ETThDataset, ETTmDataset, CustomDataset, STDataset, mySTDataset
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': ETThDataset,
    'ETTh2': ETThDataset,
    'ETTm1': ETTmDataset,
    'ETTm2': ETTmDataset,
    'electricity': CustomDataset,
    'exchange_rate': CustomDataset,
    'traffic': CustomDataset,
    'weather': CustomDataset,
    'traffic': CustomDataset,
    'weather': CustomDataset,
    'wind': CustomDataset,
    'METR-LA': STDataset,
    'PEMS-BAY': STDataset,
    #'Xian': CustomDataset,
    'Xian': mySTDataset
}

def load_data(args: Namespace):
    _Dataset = data_dict[args.dataset]
    train_dataset = _Dataset(args, flag='train')
    val_dataset = _Dataset(args, flag='val')
    test_dataset = _Dataset(args, flag='test')
    
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True,
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=True,
    )
    test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=True,
    )

    return train_loader, val_loader, test_loader, test_dataset
