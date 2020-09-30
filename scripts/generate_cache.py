from data_utils import SequenceLabelingDataset
from train_static_context import GENIA, GENIA_FULL, LITBANK, LITBANK_FULL, ONTONOTES
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os, subprocess


datasets = {"genia": GENIA,
            "genia_full": GENIA_FULL,
            "litbank": LITBANK, 
            "litbank_full": LITBANK_FULL, 
            "ontonotes": ONTONOTES}

DELTE_CACHE = True


if DELTE_CACHE:
    for name, dataset in datasets.items():
        for split in ['train_dir', 'dev_dir', 'test_dir']:
            cache_dir = os.path.join(dataset[split], 'cache')
            if os.path.exists(cache_dir):
                print("deleting cache:", cache_dir)
                subprocess.call(['rm', '-rf', cache_dir])

for name, dataset in datasets.items():
    for tok_type in ['bert-base-cased', 'bert-base-uncased']:

        tokenizer = AutoTokenizer.from_pretrained(tok_type, do_lower_case='uncased' in tok_type, do_basic_tokenize=False)

        for split in ['train_dir', 'dev_dir', 'test_dir']:

            print("generating cache for: ", name, tok_type, split)

            documents = SequenceLabelingDataset(dataset[split], tokenizer)
            data_loader = DataLoader(documents, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x)
            
            for doc in data_loader:
                continue