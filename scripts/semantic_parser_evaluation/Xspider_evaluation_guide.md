# Xspider Evaluation
To evaluate Xspider, you will need to download several datasets.

## Setup Instructions
To evaluate TS (test-suite accuracy), download the Spider database. Note that this database is identical to the one used in Xspider, except for the Vietnamese dataset.

- Download the Spider database from [here](https://yale-lily.github.io/spider).
- Unzip the downloaded file and place the "database" folder in the root directory of this repository.

Then, move dev.json and, train_spider.json to the data/xspider directory. These files will correspond to the original_dev_filepath argument.

## CSpider Data for Xspider
For the CSpider split in Xspider, we provide pre-processed data in the data/xspider/cspider directory. The files included are:

./data/xspider/cspider/dev.json
./data/xspider/cspider/dev_cspider_seq2seq.json
./data/xspider/cspider/zh_dev.json
./data/xspider/cspider/dev_cspider_zh_seq2seq.json
Files containing "zh" are used for evaluation on the "zh" split. (not zh-full.)

We thank the authors of [Cspider](https://github.com/taolusi/chisp) for providing these datasets. Please ensure you cite the original paper if you use this data.

## VSpider Data
For evaluation on VSpider, please email deokhk@postech.ac.kr, as the data cannot be redistributed directly.

