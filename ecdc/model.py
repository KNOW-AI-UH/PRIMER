from numpy import NaN
import torch
import os
from collections import OrderedDict
from transformers import Adafactor
from tqdm import tqdm

import pandas as pd
import pdb
from datasets import load_dataset, load_metric
import json
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LEDConfig,
    LEDTokenizer,
    LEDForConditionalGeneration,
)
from dataloader import (
    get_dataloader_summ,
    get_dataloader_summiter,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss

class PRIMERSummarizer(pl.LightningModule):
    def __init__(self, args):
        super(PRIMERSummarizer, self).__init__()
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.primer_path)
        # self.model = LEDForConditionalGeneration.from_pretrained(args.primer_path)
        self.model = self.__transfer_state_dict(args.primer_path)
        self.model.gradient_checkpointing_enable()
        # if args.debug_mode:
        #     pdb.set_trace()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_ddp = args.multi_gpu
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")

    def __transfer_state_dict(self, model_path):
        states = torch.load(model_path)
        new_states = OrderedDict()
        for k in states:
            new_k=k.replace('model','led')
            new_states[new_k]=states[k]
        new_states['led.encoder.embed_positions.weight'] = states['model.encoder.embed_positions.weight'][2:]
        new_states['led.decoder.embed_positions.weight'] = states['model.decoder.embed_positions.weight'][2:]
        new_states['lm_head.weight'] = states['model.shared.weight']
        config = LEDConfig.from_pretrained('F:\\resources\\PRIMER_multinews')
        model = LEDForConditionalGeneration(config)
        model.load_state_dict(new_states)
        return model

    def forward(self, input_ids, output_ids):
        # input_ids=batch.src
        # output_ids=batch.tgt

        # pdb.set_trace()
        decoder_input_ids = output_ids[:, :-1]

        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).cuda()
        # put global attention on <s> token

        global_attention_mask[:, 0] = 1
        if self.args.join_method == "concat_start_wdoc_global":
            global_attention_mask[input_ids == self.docsep_token_id] = 1
        outputs = self.model(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=False,
        )
        lm_logits = outputs[0]
        # pdb.set_trace()
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        return lm_logits

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids, output_ids):
        lm_logits = self.forward(input_ids, output_ids)
        labels = output_ids[:, 1:].clone()
        # pdb.set_trace()

        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        if torch.isnan(loss):
            pdb.set_trace()
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # pdb.set_trace()
        # if self.args.debug_mode:
        #     return None
        input_ids, output_ids = batch
        loss = self.shared_step(input_ids, output_ids)
        # if torch.isnan(loss):
        #     pdb.set_trace()
        #     loss = self.shared_step(input_ids, output_ids)

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size": input_ids.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        # pdb.set_trace()
        return loss

    def compute_rouge_batch(self, input_ids, output_ids, gold_str):
        scorer = load_metric("rouge")
        # pdb.set_trace()

        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).cuda()
        # put global attention on <s> token

        global_attention_mask[:, 0] = 1
        if self.args.join_method == "concat_start_wdoc_global":
            global_attention_mask[input_ids == self.docsep_token_id] = 1
        generated_ids = self.model.generate(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=3 if self.args.applyTriblck else None,
            length_penalty=self.args.length_penalty,
        )

        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        # gold_str = self.tokenizer.batch_decode(
        #     output_ids.tolist(), skip_special_tokens=True
        # )

        if self.args.mode == "test":
            if self.args.applyTriblck:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_triblck_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            else:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            idx = len(os.listdir(output_dir))
        result_batch = []
        if self.args.debug_mode:
            pdb.set_trace()
        for ref, pred in zip(gold_str, generated_str):
            # change <n> to \n
            pred = pred.replace("<n>", "\n")

            if self.args.mode == "test":
                with open(os.path.join(output_dir, "%d.txt" % (idx)), "w") as of:
                    of.write(pred)
                idx += 1

            s = scorer.compute(
                predictions=[pred],
                references=[ref],
                use_agregator=False,
                use_stemmer=True,
            )
            result_batch.append(
                (
                    s["rouge1"][0].recall,
                    s["rouge1"][0].precision,
                    s["rouge1"][0].fmeasure,
                    s["rouge2"][0].recall,
                    s["rouge2"][0].precision,
                    s["rouge2"][0].fmeasure,
                    s["rougeL"][0].recall,
                    s["rougeL"][0].precision,
                    s["rougeL"][0].fmeasure,
                    s["rougeLsum"][0].recall,
                    s["rougeLsum"][0].precision,
                    s["rougeLsum"][0].fmeasure,
                )
            )

            # pdb.set_trace()
        return result_batch

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        if self.args.mode=='pretrain':
            input_ids, output_ids = batch
        else:
            input_ids, output_ids, tgt = batch
        loss = self.shared_step(input_ids, output_ids)
        if self.args.compute_rouge:
            result_batch = self.compute_rouge_batch(input_ids, output_ids, tgt)
            return {"vloss": loss, "rouge_result": result_batch}
        else:
            return {"vloss": loss}

    def compute_rouge_all(self, outputs, output_file=None):
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc["avg_score"] = avg
        if output_file:
            csv_name = (
                self.args.model_path
                + output_file
                + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
            )
            rouge_results.to_csv(csv_name)

        avgr = (avg[2] + avg[5] + avg[8]) / 3
        metrics = avg
        print("Validation Result at Step %d" % (self.global_step))
        print(
            "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, \
            Rouge-Lsum f-score: %f"
            % (metrics[9], metrics[10], metrics[11])
        )
        return names, metrics, avgr

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_rouge:
            names, metrics, avgr = self.compute_rouge_all(outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log("avgr", avgr)
            return {
                "avg_val_loss": vloss,
                "avgr": avgr,
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        tloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
        output_file = "test_%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        names, metrics, avgr = self.compute_rouge_all(outputs, output_file=output_file)
        metrics = [tloss, avgr] + metrics
        names = ["tloss", "avgr"] + names
        logs = dict(zip(*[names, metrics]))
        self.logger.log_metrics(logs, step=self.global_step)
        self.log("avgr", avgr)
        # self.log_dict(logs)
        return {"avg_test_loss": tloss, "avgr": avgr, "log": logs, "progress_bar": logs}