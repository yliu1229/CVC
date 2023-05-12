## Process data

This folder has some tools to pre-process UCF101 and HMDB51 datasets ONLY. 

For Kinetics, we use 'decord' to process video on the fly, please see ./Train/dataset_3d

### 1. Download

Download the videos from source: 
[UCF101 source](https://www.crcv.ucf.edu/data/UCF101.php), 
[HMDB51 source](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads), 
[Kinetics400 source](https://deepmind.com/research/publications/kinetics-human-action-video-dataset).

Make sure datasets are stored as follows: 

* UCF101
```
{your_path}/UCF101/videos/{action class}/{video name}.avi
{your_path}/UCF101/splits_classification/trainlist{01/02/03}.txt
{your_path}/UCF101/splits_classification/testlist{01/02/03}}.txt
```

* HMDB51
```
{your_path}/HMDB51/videos/{action class}/{video name}.avi
{your_path}/HMDB51/split/testTrainMulti_7030_splits/{action class}_test_split{1/2/3}.txt
```

### 2. Extract frames

Edit path arguments in `main_*()` functions, and `python extract_frame.py`. Video frames will be extracted. 

### 3. Collect all paths into csv

Edit path arguments in `main_*()` functions, and `python write_csv.py`. csv files will be stored in `data/` directory.








