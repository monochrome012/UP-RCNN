import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from up_rcnn.data import CopyPasteDataset
from up_rcnn.model import UP_RCNN
from utils import (average_precision, draw, get_logger, move_device,
                   setup_args, setup_seed, update)


def main():

    setup_seed(0)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    args = setup_args()

    if local_rank == 0:
        logger = get_logger(args)
        for k, v in args.__dict__.items():
            logger.info(f"{k}: {v}")

    model = UP_RCNN().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: epoch / args.warm_up_epochs
        if epoch <= args.warm_up_epochs else 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    scaler = GradScaler()

    outputs = dict()
    start_epoch = 1

    if args.checkpoint_file is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        parameters = torch.load(args.checkpoint_file, map_location=map_location)
        model.load_state_dict(parameters["model"])
        optimizer.load_state_dict(parameters["optimizer"])
        scheduler.load_state_dict(parameters["scheduler"])
        scaler.load_state_dict(parameters["scaler"])
        outputs = parameters["outputs"]
        start_epoch = parameters["epoch"] + 1

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_dataset = CopyPasteDataset(dir=args.train_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=train_dataset.collate_fn,
                              sampler=train_sampler,
                              shuffle=False,
                              drop_last=True)

    val_dataset = CopyPasteDataset(dir=args.val_dir)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=val_dataset.collate_fn,
                            sampler=val_sampler,
                            shuffle=False,
                            drop_last=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model = model.train()

        epoch_outputs = dict()

        t = time.time()
        for i, batch_inputs in enumerate(train_loader):

            batch_inputs = move_device(batch_inputs, torch.device("cuda"))

            with autocast():
                batch_outputs = model(batch_inputs)
                loss = torch.stack([v * args.loss_weight[k] for k, v in batch_outputs.items() if k in args.loss_weight]).sum()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if local_rank == 0:

                batch_outputs = {k: v.item() for k, v in batch_outputs.items() if k in args.loss_weight}
                update(epoch_outputs, batch_outputs)

                if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                    draw(epoch_outputs, args.plt_dir, args.name + "_" + str(epoch))
                    avg_time = (time.time() - t) / (i + 1)
                    logger.info("*" * 41)
                    logger.info(f"Epoch {epoch}    [{i+1}/{len(train_loader)}]    {avg_time}s/it")
                    logger.info("*" * 41)
                    for k, v in epoch_outputs.items():
                        logger.info(f"{k}: {np.mean(v)}")

        if local_rank == 0:
            epoch_outputs["ap"] = []

        model = model.eval()
        with torch.inference_mode():
            for i, batch_inputs in enumerate(val_loader):

                batch_inputs = move_device(batch_inputs, torch.device("cuda"))
                batch_outputs = model(batch_inputs)

                if local_rank == 0:
                    boxes_q = [input["boxes_q"].cuda() for input in batch_inputs]
                    ap = average_precision(batch_outputs["pred_boxes"], batch_outputs["pred_scores"], boxes_q)
                    epoch_outputs["ap"].append(ap.item())

        scheduler.step()

        if local_rank == 0:
            logger.info(f"Average Precision: {np.mean(epoch_outputs['ap'])}")

            update(outputs, epoch_outputs)
            draw(outputs, args.plt_dir, args.name)

            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "outputs": outputs,
                    "epoch": epoch,
                }, os.path.join(args.checkpoint_dir,
                                str(epoch) + ".pth.tar"))


if __name__ == "__main__":
    main()
