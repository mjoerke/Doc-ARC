# docattention

## Models and data

|Model|Path|
|---|---|
|LSTM|`scripts/bert-pytorch-sequence-labeling-lstm`|
|DocAttention|`scripts/attention_seq_prelstm`|

|Data|Path|Description|
|---|---|---|
|OntoNotes|`data/ontonotes/ontonotes_ner_data`|OntoNotes NER docs with >1K tokens
|GENIA|`data/genia/genia_ner_data`|JNLPBA data, subsetted to documents where we were able to find the full text article
|LitBank|`data/litbank/flat_litbank`|All LitBank docs (from NAACL data)
|GENIA Full|`data/genia/genia_ner_data_full`|GENIA data above, with full document context appended to each trainin/dev/test/ doc. (Still in tsv format, with "NONE" label for extra context)
|LitBank Full|`data/litbank/flat_litbank_full`|LitBank data above, with full document context appended to each trainin/dev/test/ doc. (Still in tsv format, with "NONE" label for extra context)


## Training

`cd` to the model path, set the correct `DATA` variable and execute the following code to train and test for each dataset.  You'll want to set the model and log files too to make sure they're not clobbered.

### OntoNotes


```sh
# Train
python3 sequence_labeling.py --trainFolder $DATA/train --devFolder $DATA/development --mode train --modelFile ontonotes.ner.model --tagFile $DATA/ontonotes.tagset --metric span_fscore > logs/ontonotes_train.log 2>&1

# Test
python3 sequence_labeling.py --testFolder $DATA/test --mode test --modelFile ontonotes.ner.model --tagFile $DATA/ontonotes.tagset --metric span_fscore > logs/ontonotes_test.log 2>&1
```

### GENIA

```sh
# Train
python3 sequence_labeling.py --trainFolder $DATA/train --devFolder $DATA/dev --mode train --modelFile genia.ner.model --tagFile $DATA/genia.tagset --metric span_fscore > logs/genia_train.$i.log 2>&1

# Test
python3 sequence_labeling.py --testFolder $DATA/test --mode test --modelFile genia.ner.model --tagFile $DATA/genia.tagset --metric span_fscore > logs/genia_test.$i.log 2>&1
```

### LitBank

```sh
# Train
python3 sequence_labeling.py --trainFolder $DATA/train --devFolder $DATA/dev --mode train --modelFile litbank.ner.$i.model --tagFile $DATA/litbank.tagset --metric span_fscore > logs/litbank_train.$i.log 2>&1

# Test
python3 sequence_labeling.py --testFolder $DATA/test --mode test --modelFile litbank.ner.$i.model --tagFile $DATA/litbank.tagset --metric span_fscore > logs/litbank_test.$i.log 2>&1
```
