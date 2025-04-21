# BioNNE-L

## Step1: Retrieval

### English

#### Dev

|                    Model                    | Acc@1  | Acc@5  | Acc@10 |
| :-----------------------------------------: | :----: | :----: | :----: |
|               gebert_eng_gat                | 0.5898 | 0.7654 | 0.7979 |
|              BioLinkBERT-large              | 0.427  | 0.603  |   \    |
|              BioLinkBERT-base               | 0.472  | 0.653  | 0.671  |
|      SapBERT-from-PubMedBERT-fulltext       | 0.6115 | 0.7698 | 0.8043 |
| SapBERT-from-PubMedBERT-fulltext-mean-token | 0.6038 | 0.7723 | 0.8184 |


#### Train
|                    Model                    | Acc@1  | Acc@5  | Acc@10 |
| :-----------------------------------------: | :----: | :----: | :----: |
|               gebert_eng_gat                | 0.3732 | 0.6494 | 0.7431 |
|      SapBERT-from-PubMedBERT-fulltext       | 0.3595 | 0.6509 | 0.7513 |
| SapBERT-from-PubMedBERT-fulltext-mean-token | 0.3587 | 0.6349 | 0.7253 |


### Russian

#### Dev
|                    Model                     | Acc@1  | Acc@5  | Acc@10 |
| :------------------------------------------: | :----: | :----: | :----: |
|    SapBERT-UMLS-2020AB-all-lang-from-XLMR    | 0.4914 | 0.5497 | 0.5686 |
| SapBERT-UMLS-2020AB-all-lang-from-XLMR-large | 0.5103 | 0.5613 | 0.5763 |



## Step2: Rank

### English

## retrieval data

|                        Model                        |  Acc   | Base on(Acc@1/Acc@5/Acc@10) |     Approach     |              P.S.               |
| :-------------------------------------------------: | :----: | :-------------------------: | :--------------: | :-----------------------------: |
|                   gebert_eng_gat                    |   \    |              \              |    Multilabel    |                \                |
|                  BioLinkBERT-base                   | 0.5441 |    0.5285/0.7658/0.8011     |    Multilabel    |                \                |
|                  BioLinkBERT-large                  | 0.5269 |    0.5285/0.7658/0.8011     |    Multilabel    |                \                |
|     BiomedNLP-BiomedBERT-base-uncased-abstract      | 0.5441 |    0.5285/0.7658/0.8011     |    Multilabel    |                \                |
|                  bart-base-uncased                  | 0.5253 |    0.5285/0.7658/0.8011     |    Multilabel    |                \                |
| BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext |   \    |              \              |    regression    |                \                |
| BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext | 0.6536 |    0.6115/0.7698/0.8043     | 2-Classification | data: NEREL, retrieval_topk: 5  |
|     BiomedNLP-BiomedBERT-base-uncased-abstract      | 0.6604 |    0.6115/0.7698/0.8043     | 2-Classification | data: NEREL, retrieval_topk: 5  |
| BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext | 0.6532 |    0.6115/0.7698/0.8043     | 2-Classification | data: NEREL, retrieval_topk: 10 |



# Extra Dataset
|        dataset         | Performance |
| :--------------------: | :---------: |
| nerel-ds/NEREL(github) |      \      |
|                        |             |