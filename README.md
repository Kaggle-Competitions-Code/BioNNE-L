# BioNNE-L

## Step1: Retrieval

### English

|              Model               | Acc@1  | Acc@5  | Acc@10 |
| :------------------------------: | :----: | :----: | :----: |
|          gebert_eng_gat          | 0.566  | 0.799  | 0.834  |
|        BioLinkBERT-large         | 0.427  | 0.603  |   \    |
|         BioLinkBERT-base         | 0.472  | 0.653  | 0.671  |
| SapBERT-from-PubMedBERT-fulltext | 0.6115 | 0.7698 | 0.8043 |



## Step2: Rank

## retrieval data

|                        Model                        |  Acc   | Base on(Acc@1/Acc@5/Acc@10) |  Approach  |
| :-------------------------------------------------: | :----: | :-------------------------: | :--------: |
|                   gebert_eng_gat                    |        |                             |            |
|                  BioLinkBERT-base                   | 0.5441 |    0.5285/0.7658/0.8011     | Multilabel |
|                  BioLinkBERT-large                  | 0.5269 |    0.5285/0.7658/0.8011     | Multilabel |
|     BiomedNLP-BiomedBERT-base-uncased-abstract      | 0.5441 |    0.5285/0.7658/0.8011     | Multilabel |
|                  bart-base-uncased                  | 0.5253 |    0.5285/0.7658/0.8011     | Multilabel |
| BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext |   \    |                             | regression |