# Data files

The repository does not redistribute third-party datasets. Prepare the four
datasets used in the paper and place them here:

```text
data/ML-1M.txt
data/Digital_Music.txt
data/Musical_Instruments.txt
data/Office_Products.txt
```

Each line must have the following form, with interactions sorted from oldest to
newest:

```text
user_id item_1 item_2 item_3 ...
```

Use positive consecutive item IDs. ID `0` is reserved for padding. Every user
must have at least three interactions for the leave-one-out split.
