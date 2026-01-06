# lance-huggingface
LanceDB huggingface datasets


## Data organization

You can either include a `README.md` with [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards)
directly in a raw Lance dataset, or adopt a Hugging Face–style directory structure,
placing Lance datasets for different splits in the `/data` directory:

```sh
my_dataset/
├── README.md
└── data/
    ├── train.lance
    ├── test.lance
    └── validation.lance
```

