|Data|Path|Description|
|---|---|---|
|OntoNotes|`data/ontonotes/ontonotes_ner_data`|OntoNotes NER docs with >1K tokens
|GENIA|`data/genia/genia_ner_data`|JNLPBA data, subsetted to documents where we were able to find the full text article
|GENIA Full|`data/genia/genia_ner_data_full`|JNLPBA data, with full document context appended to each train/dev/test/ document. (Still in tsv format, with "NULL" label for extra context)
|LitBank|`data/litbank/flat_litbank`|All LitBank docs (from NAACL data)
|LitBank Full|`data/litbank/flat_litbank_full`|LitBank data above, with full document context appended to each train/dev/test/ document. (Still in tsv format, with "NULL" label for extra context)
