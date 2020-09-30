# JNLPBA

**To process the original GENIA data from scratch:**

Extract from original GENIA data just the annotations from articles we have pdfs for; save each article to its own file.

```sh
python ../scripts/filterGENIA.py pdf_list.train.txt original/Genia4ERtask2.iob2 genia_ner_data/train

python ../scripts/filterGENIA.py pdf_list.dev.txt original/Genia4ERtask2.iob2 genia_ner_data/dev

python ../scripts/filterGENIA.py pdf_list.test.txt original/Genia4EReval2.iob2 genia_ner_data/test
```

Original data source:

Train/dev: `Genia4ERtask2.iob2` in 
http://www.nactem.ac.uk/tsujii/GENIA/ERtask/Genia4ERtraining.tar.gz
	
Test: `Genia4EReval2.iob2` in 
http://www.nactem.ac.uk/tsujii/GENIA/ERtask/Genia4ERtest.tar.gz

### Process pdf data

We extract the text from all pdfs using OCR; post-process the OCR output to deal with end-line hyphenization, etc.

```sh
for i in `ls articles/raw_ocr`
do
python ../scripts/OCR_postprocess.py articles/raw_ocr/$i > articles/processed_texts/$i
done
```

### Append document context

```sh
python3 ../scripts/add_full_text_GENIA.py genia_ner_data/train genia_ner_data_full/train 
python3 ../scripts/add_full_text_GENIA.py genia_ner_data/dev genia_ner_data_full/dev 
python3 ../scripts/add_full_text_GENIA.py genia_ner_data/test genia_ner_data_full/test 
```

### Create tagsets

```sh
python3 ../scripts/create_taglist.py genia_ner_data > genia_ner_data/genia.tagset
python3 ../scripts/create_taglist.py genia_ner_data/genia_ner_data_full/genia.tagset
```