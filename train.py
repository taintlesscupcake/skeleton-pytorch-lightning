import argparse
import module
from pytorch_lightning import trainer
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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Batch size for training')
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

    network_module = module.SkeletonModule(args)
    data_module = module.SkeletonDataModule(args)

    if args.ckpt != "":
        network_module.load_state_dict(args.ckpt)
        print("Load ckpt from {}".format(args.ckpt))

    # Make your own checkpoint callback.
    checkpointer = ModelCheckpoint(dirpath=f"logs/{args.name}",
                                   filename='{epoch}-{val_loss:.2f}',
                                   monitor='val_loss',
                                   save_last=True,
                                   save_weights_only=False,
                                   mode='min')

    # Tensorboard logger callback.
    logger = TensorBoardLogger("logs", name=args.name)

    # Make your own trainer. This setting will use all GPUs available. 
    # If you want to use specific GPUs, use CUDA_VISIBLE_DEVICES before running the script.
    trainer = trainer.Trainer(accelerator='gpu',
                              devices="auto",
                              strategy="ddp_find_unused_parameters_false",
                              max_epochs=args.epochs,
                              callbacks=[checkpointer],
                              logger=logger)

    trainer.fit(network_module, data_module)