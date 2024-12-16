import numbers
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openrlhf.datasets.utils import zero_pad_sequences


class ProcessRewardDataset(Dataset):
    """
    Dataset for process reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.step_key = getattr(self.strategy.args, "step_key", None)
        self.step_label_key = getattr(self.strategy.args, "step_label_key", None)

        # Store the processed data in class attributes
        self.question_list = dataset[self.input_key]
        self.steps_list = dataset[self.step_key]
        self.step_scores_list = dataset[self.step_label_key]

    def __len__(self):
        length = len(self.question_list)
        return length

    def __getitem__(self, idx):
        question = self.question_list[idx] + "\n\n"
        steps = self.steps_list[idx]
        steps = [step + "\n\n" for step in steps[:-1]] + steps[-1:]
        steps_score = self.step_scores_list[idx]

        question_with_steps = [question] + steps
        full_text = "".join(question_with_steps)
        input_token = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"][0]
        input_len = input_ids.shape[-1]

        step_end_idx = []
        scores = []
        question_len = len(self.tokenizer.encode(question, add_special_tokens=False))
        cumlen = question_len
        for step_response, score in zip(steps, steps_score):
            step_token = self.tokenizer.encode(
                step_response,
                add_special_tokens=False,
            )
            step_len = len(step_token)
            cumlen += step_len
            if cumlen <= input_len:
                step_end_idx.append(cumlen)
                scores.append(score)                

        labels = torch.full_like(input_ids, -100, dtype=torch.float)
        step_end_idx = torch.LongTensor(step_end_idx)
        labels[step_end_idx - 1] = torch.tensor(scores, dtype=torch.float)

        return (
            input_ids,
            input_token["attention_mask"],
            labels,
        )

    def collate_fn(self, item_list):
        input_ids = []
        input_masks = []
        label_ids = []
        for input_id, input_mask, label_id in item_list:
            input_ids.append(input_id)
            input_masks.append(input_mask)
            label_ids.append(label_id)

        padding_side = "right"
        input_ids = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks, side=padding_side)
        label_ids = zero_pad_sequences(label_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        return input_ids, input_masks, label_ids

    def packing_collate_fn(self, item_list):
        input_ids = []
        input_att_masks = []
        input_seq_lens = []
        label_ids = []
        index = 1
        for input_id, input_mask, label_id in item_list:
            input_ids.append(input_id.flatten())
            input_att_masks.append(torch.full_like(input_id.flatten(), index))
            input_seq_lens.append(len(input_id.flatten()))

            label_ids.append(label_id.flatten())
            index += 1

        packed_input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(input_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = input_seq_lens
        packed_label_ids = torch.cat(label_ids, dim=0).unsqueeze(0)

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, packed_label_ids
