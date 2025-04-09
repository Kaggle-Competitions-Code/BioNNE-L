# BioNNE-L

## Step1: Retrieval

|       Model       | Acc@1 | Acc@5 | Acc@10 |
| :---------------: | :---: | :---: | :----: |
|  gebert_eng_gat   | 0.566 | 0.799 | 0.834  |
| BioLinkBERT-large | 0.427 | 0.603 |   \    |
| BioLinkBERT-base  | 0.472 | 0.653 | 0.671  |



## Step2: Rank

## retrieval data
train: Acc: 0.3747, Acc@2: 0.5067, Acc@5: 0.6952, Acc@10: 0.8059
dev: Acc: 0.5285, Acc@2: 0.6784, Acc@5: 0.7658, Acc@10: 0.8011


|                   Model                    |      Approach       |  Acc   |
| :----------------------------------------: | :-----------------: | :----: |
|               gebert_eng_gat               | Classification-only |        |
|              BioLinkBERT-base              |                     | 0.5441 |
|             BioLinkBERT-large              |                     | 0.5269 |
| BiomedNLP-BiomedBERT-base-uncased-abstract |                     | 0.5441 |
|             bart-base-uncased              |                     | 0.5253 |
