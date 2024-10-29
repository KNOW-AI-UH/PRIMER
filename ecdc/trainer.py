
import torch
import os
import argparse
import json
from dataloader import ECDCJSONDataset, collate_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pathlib import Path
from model import PRIMERSummarizer
from pytorch_lightning.plugins.environments import SLURMEnvironment


def train(args):
    accelerator = 'gpu' if torch.cuda.device_count() else 'cpu'
    args.compute_rouge = True
    model = PRIMERSummarizer(args)


    

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "/tb_logs", name="my_model")
    # initialize trainer
    trainer = pl.Trainer(
        devices='auto',
        num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
        accelerator=accelerator,
        max_steps=args.total_steps,
        accumulate_grad_batches=args.acc_batch,
        val_check_interval=0.25,
        # check_val_every_n_epoch=1 if args.num_train_data > 100 else 5,
        logger=logger,
        log_every_n_steps=5,
        enable_checkpointing=True,
        enable_progress_bar=False,
        precision=32,
        strategy=DDPStrategy(find_unused_parameters=False),
        plugins=[SLURMEnvironment(auto_requeue=False)],
    )

    # load datasets
    train_json = os.path.join(args.data_path, f'train_data_{args.cluster_dist}.json')
    dataset = ECDCJSONDataset(
        train_json,
        args.join_method,
        model.tokenizer,
        args.max_length_input,
        args.max_length_tgt,
        mask_num=args.mask_num,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="train",
    )
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                pin_memory=True, num_workers=args.num_workers, drop_last=True,
                                prefetch_factor=2)
    val_json = os.path.join(args.data_path, f'val_data_{args.cluster_dist}.json')
    dataset = ECDCJSONDataset(
        val_json,
        args.join_method,
        model.tokenizer,
        args.max_length_input,
        args.max_length_tgt,
        mask_num=args.mask_num,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="validation",
    )
    valid_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                pin_memory=True, num_workers=args.num_workers, drop_last=True,
                                prefetch_factor=2)
    
    # pdb.set_trace()
    trainer.fit(model, train_dataloader, valid_dataloader)
    if args.test_imediate:
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args, f'test_data_{args.cluster_dist}.json')

def test(args, json_file=None):
    if json_file is None:
        json_file = f'all_data_{args.cluster_dist}.json'
    accelerator = 'gpu' if torch.cuda.device_count() else 'cpu'
    args.compute_rouge = True
    # initialize trainer
    trainer = pl.Trainer(
        devices='auto',
        num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
        accelerator=accelerator,
        max_steps=args.total_steps * args.acc_batch,
        log_every_n_steps=5,
        enable_progress_bar=False,
        precision=32,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0,
        plugins=[SLURMEnvironment(auto_requeue=False)],
    )

    if args.resume_ckpt is not None:
        model = PRIMERSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = PRIMERSummarizer(args)

    # load dataset
    test_json = os.path.join(args.data_path, json_file)
    dataset = ECDCJSONDataset(
        test_json,
        args.join_method,
        model.tokenizer,
        args.max_length_input,
        args.max_length_tgt,
        mask_num=args.mask_num,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="test",
    )
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                pin_memory=True, num_workers=args.num_workers, drop_last=True,
                                prefetch_factor=2)
    # test
    trainer.test(model, test_dataloader, ckpt_path='best')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########################
    # Gneral
    parser.add_argument("--multi_gpu", action="store_true", help="number of gpus to use")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--cluster_dist", default=0.5, type=float, required=True)
    parser.add_argument(
        "--model_name", default="primer",
    )
    parser.add_argument(
        "--primer_path", type=str, default="../PRIMER/",
    )
    parser.add_argument("--join_method", type=str, default="concat_start_wdoc_global")
    parser.add_argument(
        "--debug_mode", action="store_true", help="set true if to debug"
    )
    parser.add_argument(
        "--compute_rouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )

    
    parser.add_argument("--model_path", type=str, default="./pegasus/")
    parser.add_argument("--saveTopK", default=3, type=int)
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        help="Path of a checkpoint to resume from",
        default=None,
    )

    parser.add_argument("--data_path", type=str, default="../dataset/")
    parser.add_argument("--dataset_name", type=str, default="ecdc")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for dataloader",
    )

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--adafactor", action="store_true", help="Use adafactor optimizer"
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="seed for random sampling, useful for few shot learning",
    )

    ########################
    # For training
    parser.add_argument(
        "--pretrained_model_path", type=str, default="./pretrained_models/",
    )
    parser.add_argument(
        "--limit_valid_batches", type=int, default=None,
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--accum_data_per_step", type=int, default=16, help="Number of data per step"
    )
    parser.add_argument(
        "--total_steps", type=int, default=50000, help="Number of steps to train"
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=-1,
        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
    )

    parser.add_argument(
        "--fix_lr", action="store_true", help="use fix learning rate",
    )
    parser.add_argument(
        "--test_imediate", action="store_true", help="test on the best checkpoint",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="whether this is a run for few shot learning",
    )
    ########################
    # For testing
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Number of batches to test in the test mode.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1,
        help="length penalty of generated text",
    )
    parser.add_argument(
        "--mask_num",
        type=int,
        default=0,
        help="Number of masks in the input of summarization data",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="batch size for test, used in few shot evaluation.",
    )
    parser.add_argument(
        "--applyTriblck",
        action="store_true",
        help="whether apply trigram block in the evaluation phase",
    )

    args = parser.parse_args()  # Get pad token id
    ####################
    args.acc_batch = args.accum_data_per_step // args.batch_size
    if args.acc_batch < 1:
        args.acc_batch = 1
    args.data_path = os.path.join(args.data_path, args.dataset_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

    print(args)
    with open(
        os.path.join(
            args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)
        ),
        "w",
    ) as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    else:

        test(args)