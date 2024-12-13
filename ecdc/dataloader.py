from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import torch
from random import shuffle
import random
import os
from nltk.tokenize import sent_tokenize
import re
import sys
import json
from collections import defaultdict

def insert_strings(args):
    doc, diseases, locations = args
    positions = []
    strings = []
    for idx in diseases:
        positions.append(idx[0])
        strings.append('<DISEASE>')
        positions.append(idx[1])
        strings.append('</DISEASE>')
    for idx in locations:
        positions.append(idx[0])
        strings.append('<LOCATION>')
        positions.append(idx[1])
        strings.append('</LOCATION>')
    
    doc = list(doc)
    for i, pos in enumerate(positions):
        doc.insert(pos+i, strings[i])
    return ''.join(doc)


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    if batch[0][0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0][0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        assert False
    train = True
    if len(batch[0]) == 3:
        train = False
        tgt = [item[2] for item in batch]
        batch = [item[:2] for item in batch]
    input_ids, output_ids = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    if train:
        return input_ids, output_ids
    else:
        return input_ids, output_ids, tgt

class ECDCJSONDataset(Dataset):
    def __init__(
        self,
        json_file,
        join_method,
        tokenizer,
        max_input_len,
        max_output_len,
        mask_num=5,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="train",
    ):
        dataset = self.__load_data(json_file)
        self.dataset = dataset
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len - 2
        self.max_output_len = max_output_len
        if join_method == "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.use_dl_special_tokens = len(self.tokenizer.special_tokens_map['additional_special_tokens']) > 1
            
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        if num_data != -1 and not is_test and num_data < len(list(dataset)):
            random.seed(rand_seed)
            self.hf_dataset = random.sample(list(dataset), num_data)
        self.dataset_type = dataset_type

    def __load_data(self, json_file):
        return json.loads(open(json_file, 'r', encoding='utf-8').read())
        # dataset = {}
        # for doc in json.loads(open(json_file, 'r', encoding='utf-8').read()):
        #     key = doc['bulletpoint_text']
        #     if key not in dataset:
        #         summary = re.sub(r"^\d+\.\s+", '', doc['bulletpoint_text'].strip())
        #         dataset[key] = {
        #             "document": [],
        #             "summary": summary
        #         }
        #     dataset[key]['document'].append(doc['news_content'])
        # return list(dataset.values())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        all_docs = entry["document"]
        # all_docs = [_ for _ in entry['cluster'] if type(_) == str] + entry["document"]
        if self.use_dl_special_tokens:
            all_docs = list(map(insert_strings, zip(entry["document"], entry['diseases'], entry['locations'])))
        tgt = entry["summary"]
        
        
        if self.join_method == "plain_concat":
            src = "\n".join(all_docs)
            input_ids = self.tokenizer.encode(
                src, truncation=True, max_length=self.max_input_len
            )
        elif self.join_method == "concat_start_eachdoc":
            input_text = []
            for doc in all_docs:
                length = 0
                all_sents = sent_tokenize(doc)
                for s in all_sents:
                    input_text.append(s)
                    length += len(s.split())
                    if length >= self.max_input_len // len(all_docs):
                        break
            input_ids = self.tokenizer.encode(
                " ".join(input_text),
                truncation=True,
                max_length=self.max_input_len,
            )
        elif self.join_method == "concat_start_eachdoc_wsent_global":
            input_ids = []
            for doc in all_docs:
                sents = [
                    " [sent] ".join(sent_tokenize(p)) + " [sent]"
                    for p in doc.split("\n")
                    if p != ""
                ]
                doc = "\n".join(sents)
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=self.max_input_len // len(all_docs),
                    )[1:-1]
                )
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
            )
        elif self.join_method == "concat_start_wdoc_global":
            mask_num = self.mask_num

            input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
            for doc in all_docs:
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=(self.max_input_len - mask_num) // len(all_docs),
                    )[1:-1]
                )
                input_ids.append(self.docsep_token_id)
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
            )

        output_ids = self.tokenizer.encode(
            tgt, truncation=True, max_length=self.max_output_len
        )

            
        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        if self.dataset_type == "train":
            return torch.tensor(input_ids), torch.tensor(output_ids)
        else:
            return torch.tensor(input_ids), torch.tensor(output_ids), tgt