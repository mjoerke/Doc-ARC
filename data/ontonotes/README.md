# Create OntoNotes data


Clone OntoNotes-5.0-NER-BIO repo:

```sh
git clone https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO
```


Download OntoNotes and unpack to a top-level directory of `ontonotes-release-5.0`.

```sh
cd OntoNotes-5.0-NER-BIO
./conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ../ontonotes-release-5.0/data/files/data ./conll-formatted-ontonotes-5.0
```

This creates a bunch of *.conll files in the `conll-formatted-ontonotes-5.0` subdirectory.  Using the same train/development/test splits in the original (CoNLL-2012) data, convert files to BIO format and only keep files that are at least 1,000 tokens.

```sh
python3 ../../../scripts/create_ontonotes_data.py ../ontonotes-release-5.0 ontonotes_ner_data
python3 ../../../scripts/create_taglist.py ontonotes_ner_data > ontonotes_ner_data/ontonotes.tagset 
```
Splits are in `train/`, `development/` and `test/` subdirectories within `ontonotes_ner_data`.

