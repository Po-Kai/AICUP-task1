import argparse
import logging
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification
from transformers import RobertaConfig, RobertaForSequenceClassification
from transformers import XLNetConfig, XLNetForSequenceClassification


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification),
    "roberta": (RobertaConfig, RobertaForSequenceClassification),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification)
}


class F1():
    def __init__(self):
        self.threshold = 0.5
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = "F1"

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        predicts = (predicts > self.threshold).float()
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth * predicts).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20) # prevent divided by zero
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_dataloader(
    features,
    sampler,
    batch_size=8,  
    num_workers=16, 
    worker_init_fn=None
):
    all_input_ids = torch.tensor([feature["net_inputs"]["input_ids"] for feature in features], dtype=torch.long)
    all_input_mask = torch.tensor([feature["net_inputs"]["attention_mask"] for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature["net_inputs"]["token_type_ids"] for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature["label"] for feature in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids, all_label_ids)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, worker_init_fn=worker_init_fn, num_workers=num_workers)
    return dataloader


def train(
    args, 
    train_dataset, 
    eval_dataset, 
    model
):
    """ Train the model """
    # Create tensorboard writer
    train_tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "train"))
    eval_tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "valid"))


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    if args.seed >= 0:
        worker_init_fn = lambda worker_id: np.random.seed(args.seed)
    else:
        worker_init_fn = None
    train_dataloader = get_dataloader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, worker_init_fn=worker_init_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        if args.warmup_steps < 0:
            args.warmup_steps = t_total * args.warmup_proportion 

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # pos_weight = torch.tensor([1.0000, 1.4313, 0.9779, 1.1343, 2.5133, 14.8202]).to(args.device)
    # criteria = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criteria = nn.BCEWithLogitsLoss()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    if args.seed >= 0:
        set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        train_f1_score = F1()
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]
            }
            labels = batch[3].float()
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            logits = outputs[0]
            loss = criteria(logits, labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            
            preds = torch.sigmoid(logits.detach()).cpu()
            labels = labels.detach().cpu()
            train_f1_score.update(preds, labels)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    eval_results = evaluate(args, eval_dataset, model)
                    eval_tb_writer.add_scalar("loss", eval_results["loss"], global_step)
                    eval_tb_writer.add_scalar("f1_score", eval_results["f1"], global_step)
                    
                    train_tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    train_tb_writer.add_scalar("loss", ( (tr_loss - logging_loss) / args.logging_steps ), global_step)
                    train_tb_writer.add_scalar("f1_score", train_f1_score.get_score(), global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    train_tb_writer.close()
    eval_tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = get_dataloader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    criteria = nn.BCEWithLogitsLoss()
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    eval_f1_score = F1()

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]
            }
            labels = batch[3].float()
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            logits = outputs[0]
            tmp_eval_loss = criteria(logits, labels)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        preds = torch.sigmoid(logits.detach()).cpu()
        labels = labels.detach().cpu()
        eval_f1_score.update(preds, labels)

    eval_loss = eval_loss / nb_eval_steps

    results = {
        "loss": eval_loss,
        "f1": eval_f1_score.get_score()
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir", default=None, type=str, required=True,
                        help="Set the logdir path for SummaryWriter")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="[bert-large-cased, xlnet-large-cased, roberta-large...]")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_proportion if max_steps=-1.")

    parser.add_argument("--logging_steps", type=int, default=25,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=25,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    # add to load part data
    parser.add_argument("--ensemble_id", type=int, default=0,
                        help="ensemble_id")
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir)
        )
    else:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger.warning("Process device: %s, n_gpu: %s, 16-bits training: %s",
                   device, args.n_gpu, args.fp16)

    # Set seed
    if args.seed >= 0:
        set_seed(args)

    # load data
    train_filename = "train_{}_{}_{}.pt".format("roberta-large", args.ensemble_id, "v2") # v2 is seg 10
    eval_filename  = "valid_{}_{}_{}.pt".format("roberta-large", args.ensemble_id, "v2")
    train_dataset = torch.load(os.path.join(args.data_dir, train_filename))
    eval_dataset = torch.load(os.path.join(args.data_dir, eval_filename))
    logger.info("Loading training dataset %s", train_filename)
    logger.info("Loading evaluation dataset %s", eval_filename)
    
    # load model
    _, model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path,
                                        num_labels=6)
    model.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)

    # training
    global_step, tr_loss = train(args, train_dataset, eval_dataset, model)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
