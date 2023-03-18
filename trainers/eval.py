import csv
import math
import argparse
import glob
import logging
import os
import random
import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertForMaskedLM,
    BertModel, BertTokenizer
)

from .args import get_args
from data_processing import data_processors, data_classes
from .mlm_utils import mask_tokens
from .train_utils import pairwise_accuracy, evaluate_standard

# Tensorboard utilities.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Accuracy metrics.
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Loggers.
logger = logging.getLogger(__name__)

##################################################
# TODO: Model Selection
# Please fill in the below to obtain the
# `AutoConfig`, `AutoTokenizer` and some auto
# model classes correctly. Check the documentation
# for essential args.

# (1) Load config
config = AutoConfig.from_pretrained(args.model_name_or_path)
#raise NotImplementedError("Please finish the TODO!")

# (2) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#raise NotImplementedError("Please finish the TODO!")

if args.training_phase == "pretrain":
# (3) Load MLM model if pretraining (Optional)
# Complete only if doing MLM pretraining for improving performance
raise NotImplementedError("Please finish the TODO!")
else:
# (4) Load sequence classification model otherwise
model = AutoModelForSequenceClassification.from_config(config)
#raise NotImplementedError("Please finish the TODO!")

# End of TODO.
##################################################

def load_and_cache_examples(args, task, tokenizer, evaluate=False,
                            data_split="test", data_dir=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = data_processors[task](data_dir=args.data_dir, args=args)

    # Getting the examples.
    if data_split == "test" and evaluate:
        examples = (processor.get_test_examples())
    elif (data_split == "val" or data_split == "dev") and evaluate:
        examples = (processor.get_dev_examples())
    elif data_split == "train" and evaluate:
        examples = (processor.get_train_examples())
    elif "test" == data_split:
        examples = (processor.get_test_examples())
    else:
        examples = (
            processor.get_test_examples()
            if evaluate else processor.get_train_examples()
        )

    logging.info("Number of {} examples in task {}: {}".format(
        data_split, task, len(examples)))

    # Defines the dataset.
    dataset = data_classes[task](examples, tokenizer,
                                 max_seq_length=args.max_seq_length,
                                 seed=args.seed, args=args)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier() 
    
    return dataset


def evaluate(args, model, tokenizer, prefix="", data_split="test"):

    # Main evaluation loop.
    results = {}
    eval_dataset = load_and_cache_examples(args, args.task_name,
                                           tokenizer, evaluate=True,
                                           data_split=data_split,
                                           data_dir=args.data_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation on split: {} {} *****".format(
        data_split, prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    labels = None
    has_label = False

    guids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)

        if not args.do_train or (args.do_train and args.eval_split != "test"):
            # guid = batch[-1].cpu().numpy()[0]
            # guids.append(guid)
            guid = list(batch[-1].cpu().numpy())
            guids += guid

        with torch.no_grad():
            # Processes a batch.
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if (args.do_train and len(batch) > 3) or (not args.do_train and len(batch) > 4):
                has_label = True
                inputs["labels"] = batch[3]

            # Clears token type ids if using MLM-based models.
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "roberta"] \
                             else None
                )  # XLM and DistilBERT don't use segment_ids
                inputs["token_type_ids"] = None

            if args.training_phase == "pretrain":
                masked_inputs, lm_labels = mask_tokens(
                    inputs["input_ids"], tokenizer, args)
                inputs["input_ids"] = masked_inputs
                inputs["labels"] = lm_labels

            ##################################################
            # TODO: Evaluation Loop
            # (1) Run forward and get the model outputs
            #raise NotImplementedError("Please finish the TODO!")
            
            if has_label or args.training_phase == "pretrain":
                outputs=model(**inputs)
                # (2) If label present or pretraining, compute the loss and prediction logits
                # Label the loss as `eval_loss` and logits as `logits`
                # Hint: See the HuggingFace transformers doc to properly get the loss
                # AND the logits from the model outputs, it can simply be 
                # indexing properly the outputs as tuples.
                # Make sure to perform a `.mean()` on the eval loss and add it
                # to the `eval_loss` variable.
                eval_loss+=outputs.loss.mean()
                logits=outputs.logits
                #raise NotImplementedError("Please finish the TODO!")
            else:
                outputs=model(**inputs)
                logits=outputs.logits
                # (3) If labels not present, only compute the prediction logits
                # Label the logits as `logits`
                #raise NotImplementedError("Please finish the TODO!")
            #logits=torch.nn.Softmax(logits)#.dim
            #s = torch.nn.Softmax()
            #logits = s(logits)
            logits = torch.nn.functional.softmax(logits, dim=1)
            #print("LOGITS", logits)
            
            # (4) Convert logits into probability distribution and relabel as `logits`
            # Hint: Refer to Softmax function
            #raise NotImplementedError("Please finish the TODO!")

            # End of TODO.
            ##################################################

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            if has_label:
                labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if has_label:
                labels = np.append(labels,
                    inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.max_eval_steps > 0 and nb_eval_steps >= args.max_eval_steps:
            logging.info("Early stopping"
                " evaluation at step: {}".format(args.max_eval_steps))
            break

    # Organize the predictions.
    preds = np.reshape(preds, (-1, preds.shape[-1]))
    preds = np.argmax(preds, axis=-1)

    if has_label or args.training_phase == "pretrain":
        # Computes overall average eavl loss.
        eval_loss = eval_loss / nb_eval_steps

        eval_loss_dict = {"{}_loss".format(args.task_name): eval_loss}
        results.update(eval_loss_dict)

        eval_perplexity = 0
        eval_acc = 0
        eval_prec = 0
        eval_recall = 0
        eval_f1 = 0
        eval_pairwise_acc = 0

        if args.training_phase == "pretrain":
            # For `pretrain` phase, we only need to compute the
            # metric "perplexity", that is the exp of the eval_loss.
            eval_perplexity = math.exp(eval_loss)
        else:
            # Standard evalution
            eval_acc, eval_prec, eval_recall, eval_f1 = evaluate_standard(preds, \
                                                labels, args.score_average_method)
            
            # Pairwise accuracy
            if args.task_name == "com2sense":
                eval_pairwise_acc = pairwise_accuracy(guids, preds, labels)

        if args.training_phase == "pretrain":
            eval_acc_dict = {"{}_perplexity".format(args.task_name): eval_perplexity}
        else:
            eval_acc_dict = {"{}_accuracy".format(args.task_name): eval_acc}
            eval_acc_dict["{}_precision".format(args.task_name)] = eval_prec
            eval_acc_dict["{}_recall".format(args.task_name)] = eval_recall
            eval_acc_dict["{}_F1_score".format(args.task_name)] = eval_f1
            # Pairwise accuracy.
            if args.task_name == "com2sense":
                eval_acc_dict["{}_pairwise_accuracy".format(args.task_name)] = eval_pairwise_acc

        results.update(eval_acc_dict)

        output_eval_file = os.path.join(args.output_dir,
            prefix, "eval_results_split_{}.txt".format(data_split))

    if has_label:
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} on split: {} *****".format(prefix, data_split))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    # Stores the prediction .txt file to the `args.output_dir`.
    if not has_label:
        pred_file = os.path.join(args.output_dir, "com2sense_predictions.txt")
        pred_fo = open(pred_file, "w")
        for pred in preds:
            pred_fo.write(str(pred)+"\n")
        pred_fo.close()
        logging.info("Saving prediction file to: {}".format(pred_file))

    return results

# Loads models onto the device (gpu or cpu).
model.to(args.device)
print(model)
args.model_type = config.model_type
checkpoint = args.model_name_or_path
ckpt_path = os.path.join(checkpoint, "pytorch_model.bin")
model.load_state_dict(torch.load(ckpt_path))
model.to(args.device)
result = evaluate(args, model, tokenizer, prefix=prefix, data_split=args.eval_split)
result = dict((k + "_{}".format(global_step), v)
                for k, v in result.items())
print(result)