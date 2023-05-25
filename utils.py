# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import shutil
from enum import Enum
from typing import Any, Dict, TypeVar

import torch
from torch import nn

__all__ = [
    "accuracy", "load_state_dict", "make_directory", "ovewrite_named_param", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter"
]

V = TypeVar("V")


def accuracy(output, target, topk=(1,)): #function that computes the accuracy over the the k top predictions for the specified vals of k
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():# wrap block of code with torch.no_grade() => no need to compute gradients within the block operations
        maxk = max(topk)#get the maximum value from topk
        batch_size = target.size(0)#get size of the target batch

        _, pred = output.topk(maxk, 1, True, True) # Get the indices of the topk predictions
        pred = pred.t()# Transpose the prediction tensor
        correct = pred.eq(target.view(1, -1).expand_as(pred))# Check if the predictions match the target

        results = []
        for k in topk:#for each value of k in the top k predictions list
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) # Calculate the correct predictions for each topk value
            results.append(correct_k.mul_(100.0 / batch_size))# Calculate the accuracy and append to results
        return results#return the resulting list


def load_state_dict(#function that loads the state dictionary
        model: nn.Module,
        model_weights_path: str,
        ema_model: nn.Module = None,
        start_epoch: int = None,
        best_acc1: float = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)#create a checkpoint containing the loaded model weights

    if load_mode == "resume":#if the load_mode is in resume mode
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()#creat a model state dictionary
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}#populate the state dictionary
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)#update the model state dictionary
        model.load_state_dict(model_state_dict)#load the model state dictionary
        # Load ema model state dict. Extract the fitted model weights
        ema_model_state_dict = ema_model.state_dict()
        ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
        # Overwrite the model weights to the current model (ema model)
        ema_model_state_dict.update(ema_state_dict)
        ema_model.load_state_dict(ema_model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()#create a new model state dictionary
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()} #populate the model state dict. with the models that fit the given conditions
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)#update the model state dict.
        model.load_state_dict(model_state_dict)#load the model state dict.

    return model, ema_model, start_epoch, best_acc1, optimizer, scheduler


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "best.pth.tar"))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "last.pth.tar"))


class Summary(Enum):#Summary type enum class
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):#a class representing an average meter
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):#method that resets the average meter
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):#update method for the average meter
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"#function that returns the summary format string
        return fmtstr.format(**self.__dict__)

    def summary(self):#function that returns the format of the summary depending on the case
        if self.summary_type is Summary.NONE:##if the summary is of NONE type
            fmtstr = ""#return the null string
        elif self.summary_type is Summary.AVERAGE:#if the summary is of type AVERAGE
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:#if the summary is of type SUM
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:#if the summary is of type COUNT
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):#class that represents a progress meter
    def __init__(self, num_batches, meters, prefix=""):#initialization method
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)# Format string for displaying batch progress
        self.meters = meters # List of meters to display
        self.prefix = prefix # Prefix string to add to the progress display

    def display(self, batch):# Display the current batch number
        entries = [self.prefix + self.batch_fmtstr.format(batch)]# Display the values of all the meters
        entries += [str(meter) for meter in self.meters] # Print the progress display with tab-separated entries
        print("\t".join(entries))

    def display_summary(self):#method that displays the summary of the progress meter
        entries = [" *"]# Start with an asterisk as the first entry
        entries += [meter.summary() for meter in self.meters]  # Display the summary of each meter
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
