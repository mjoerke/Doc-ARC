# OntoNotes

Due to licensing restrictions, we are unable to provide the OntoNotes dataset in this repo. Instead, we provide instructions and code for processing the dataset:

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
python3 ../scripts/create_ontonotes_data.py ../ontonotes-release-5.0 ontonotes_ner_data
python3 ../scripts/create_taglist.py ontonotes_ner_data > ontonotes_ner_data/ontonotes.tagset 
```

The final splits will be located in `train/`, `development/` and `test/` subdirectories within `ontonotes_ner_data`. To ensure that your splits are correct, we provide the relevant filenames in `{train,test,dev}.txt`

