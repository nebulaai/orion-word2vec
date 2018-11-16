# Orion sample AI task - single GPU - word2vec #


# Introduction #

This is a sample AI script for testing task-processing pipeline on Nebula AI Orion working node.

# Model Description #

A basic word2vec model which:

-  Download text files from http://mattmahoney.net/dc/text8.zip, load into directory _datadir_.
-  Generate a training batch for skip-gram model.
-  Build and train skip-gram model.
-  Create embedding visualization plot.

# Results #

Results of this model include:

- Checkpoints for tensorboard visualization. 
- A figure showing a subset of first 250 words in vector place. Words that are similar end up clustering nearby each other.

# Remarks # 

- The entire task-processing time (20000 steps) on Orion platform takes around 2 minutes.
- Nebula AI working node will execute the file "word2vec_basic.py", and save all results in the directory _log_, 
- Users will be able to retrieve contents from result directory, together with system execution log.
- For more details on how-to-guides for task submission, please refer to instructions on Nebula AI developer portal. 

# Execution log from Orion #

> 2018-10-03 17:35:13-INFO- Miner starts processing this task (0xd0462a041cb50383d42146F66B4C95f719859c23)
> 2018-10-03 17:35:13-INFO- starting to download task script
> 2018-10-03 17:35:14-INFO-  task script downloaded
> 2018-10-03 17:35:14-INFO- starting to install packages from requirements.txt
> 2018-10-03 17:35:20-INFO-  packages downloaded 
> 2018-10-03 17:35:20-INFO- Starting to download data resource
> 2018-10-03 17:35:30-INFO- data resource downloaded
> 2018-10-03 17:35:30-INFO- starting to execute script
> 2018-10-03 17:37:12-INFO- ========Script is executed======
> 2018-10-03 17:37:12-INFO- The results is uploaded