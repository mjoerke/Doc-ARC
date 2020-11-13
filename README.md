# Document-Attentive Representation of Context

This reposity accompanies the publication:

> Matthew Jörke, Jon Gillick, Matthew Sims, David Bamman. [Document-Attentive Representation of Context](https://www.aclweb.org/anthology/2020.findings-emnlp.330/), *Findings of EMNLP 2020*. 

We provide PyTorch implementations for all of our models along with the datasets used in our evaluation. 

## Data Preparation

We provide full datasets for LitBank and JNLPBA (listed as `genia`) in the `data` folder. Due to licensing restrictions, we are unable to provide the full OntoNotes dataset, though `data/ontonotes` contains instructions and code to generate the dataset. 

To use a custom dataset, modify `datasets.py` such that `base_dir` points to the correct folder. To pregenerate a tokenized cache for each document, run `generate_cache.py`. This will speed up training substantially.

## Training Models

We use the Huggingface [transformers](https://huggingface.co/transformers/) library throughout our code. The code has been tested for versions specified in `requirements.txt`. 

### Static Doc-ARC

```sh
python3 train_static_context.py \
    --mode train \
    --dataset {litbank_full, genia_full, ontonotes} \
    --pretrained_dir $PRETRAINED_MODEL \
    --output_dir $OUTPUT_DIR \
    --lr 0.001 \
    --gradient_accumulation_steps 1 \
    --self_attention \
    --freeze_bert \
    --num_epochs 30 \
    --lstm_dim 256 \
    --k $ATTENTION_WIDTH \
    --model_type bert-base-cased
```

### Dynamic Doc-ARC

```sh
python3 train_dynamic_context.py \
    --mode train \
    --dataset {litbank_full, genia_full, ontonotes} \
    --output_dir $OUTPUT_DIR \
    --lr 0.0001 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --self_attention \
    --vanilla \
    --num_epochs 30 \
    --lstm_dim 128 \
    --k $ATTENTION_WIDTH \
    --model_type google/bert_uncased_L-2_H-128_A-2
```

### BERT + LSTM Baselines
Parameters in brackets correspond to {dynamic, static} BERT + LSTM baselines

```sh
python3 train_no_context_lstm.py \
    --mode train \
    --dataset {litbank, genia, ontonotes} \
    --pretrained_dir $PRETRAINED_MODEL \
    --output_dir $OUTPUT_DIR \
    --lr {0.001, 0.0001} \
    --batch_size {1, 16} \
    --gradient_accumulation_steps {4, 1} \
    --self_attention \
    --freeze_bert \
    --num_epochs 30 \
    --lstm_dim {128, 256} \
    --num_lstm {1, 2} \
    --model_type {bert-base-cased, google/bert_uncased_L-2_H-128_A-2}
```

### BERT finetuning

```sh
python3 train_no_context.py \
    --mode train \
    --dataset {litbank, genia, ontonotes} \
    --pretrained_dir $PRETRAINED_MODEL \
    --output_dir $OUTPUT_DIR \
    --lr 2e-5 \
    --batch_size 16 \
    --num_epochs 10 \
    --model_type {bert-base-cased, google/bert_uncased_L-2_H-128_A-2}
```

### Task-adaptive pretraining

Task-adaptive pretraining was run on Google Cloud with BERT's pretraining code: https://github.com/google-research/bert/

```sh
python create_pretraining_data.py \
	--input_file=$INPUT_FILE \
	--output_file=$OUTPUT_FILR \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--do_lower_case=False \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--dupe_factor=5 \
	--do_whole_word_mask=True \
	--short_seq_prob=0 \
	--masked_lm_prob=0.15 \

python run_pretraining.py \
	--input_file=$INPUT_FILE \
	--output_file=$OUTPUT_FILR \
	--do_train=True \
	--do_eval=True \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	--train_batch_size=32 \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--num_train_steps={291036,270903,25591} \ # 100 epochs for {litbank,genia,ontonotes}
	--num_warmup_steps= #0.06 * num_train_steps \
	--learning_rate=2e-5 \
	--use_tpu=True
```
