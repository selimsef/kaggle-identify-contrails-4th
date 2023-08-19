import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from training.dataset_seq import DatasetSinglePseudoRoundAll
from validation import SegEvaluator

import warnings

from training import augmentation_sets, dataset_seq
from training.augmentation_sets import WarmingAugmentations

from training.config import load_config

warnings.filterwarnings("ignore")
import argparse

from training.trainer import TrainConfiguration, PytorchTrainer


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/v2s.json")
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='1', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='')
    arg('--data-dir', type=str, default="/mnt/md0/datasets/warming/")
    arg('--folds-csv', type=str, default="folds.csv")
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--fp16', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local-rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg('--pred-dir', type=str, default="../oof")
    arg("--val", action='store_true', default=False)
    arg("--freeze-bn", action='store_true', default=False)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    conf = load_config(args.config)

    augmentations = augmentation_sets.__dict__[conf["augmentations"]]()  # type: WarmingAugmentations
    dataset_type = dataset_seq.__dict__[conf["dataset"]["type"]]
    params = conf["dataset"].get("params", {})
    print(f"Using augmentations: {augmentations.__class__.__name__} with Dataset: {dataset_type.__name__}")
    train_dataset = dataset_type(mode="train",
                                 fold=args.fold,
                                 folds_csv=args.folds_csv,
                                 dataset_dir=args.data_dir,
                                 transforms=augmentations.get_train_augmentations(conf), **params)
    val_dataset = DatasetSinglePseudoRoundAll(mode="val", dataset_dir=args.data_dir,
                                              transforms=augmentations.get_val_augmentations(conf))
    return train_dataset, val_dataset


def main():
    args = parse_args()
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        fp16=args.fp16,
        freeze_bn=args.freeze_bn
    )

    data_train, data_val = create_data_datasets(args)
    seg_evaluator = SegEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=seg_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)
    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == '__main__':
    main()
