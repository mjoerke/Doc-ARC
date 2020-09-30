# Litbank

**To process the original LitBank from scratch:**

Clone original LitBank repo

```sh
git clone https://github.com/dbamman/litbank.git
```

Extract outer entities

```sh
python ../scripts/create_litbank_data.py litbank/entities/tsv/ flat_litbank
```

