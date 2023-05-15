import argparse
import module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='unspecified',
                        type=str, help='name of experiment')
    parser.add_argument('--trainset', default='unspecified',
                        type=str, help='name of trainset')
    parser.add_argument('--testset', default='unspecified',
                        type=str, help='name of testset')
    parser.add_argument('--train_data_path', default='',
                        type=str, help='Root path for train data')
    parser.add_argument('--test_data_path', default='',
                        type=str, help='Root path for test data')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for data loader')
    parser.add_argument('--ckpt', default='', type=str,
                        help='filepath of weights')
    parser.add_argument('--description', default='',
                        type=str, help='description of experiment')
    ### Add More Arguments You Need ###

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Make your own trainer. This setting will use all GPUs available.
    # If you want to use specific GPUs, use CUDA_VISIBLE_DEVICES before running the script.
    trainer = Trainer(accelerator='gpu',
                      devices="auto")

    network_module = module.DefaultModule.load_from_checkpoint(
        checkpoint_path=args.ckpt)

    data_module = module.DefaultModule(args)

    network_module.args = args

    trainer.test(network_module, datamodule=data_module)

    print("Test Done!")
