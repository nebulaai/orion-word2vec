# Orion sample AI task - single GPU - word2vec #


# Introduction #

This is a sample AI script for testing task-processing pipeline on Nebula AI Orion working node.
Original article: https://medium.com/deep-math-machine-learning-ai/chapter-9-2-nlp-code-for-word2vec-neural-network-tensorflow-544db99f5334

# Model Description #

A basic word2vec model which:

-  Download text files from http://mattmahoney.net/dc/text8.zip, load into directory _datadir_.
-  Generate a training batch for skip-gram model.
-  Build and train skip-gram model.
-  Create embedding visualization plot.

# How to run AI model on Orion
- Step one: Packup your AI model into a zip by [Orion-Script-Converter](https://github.com/nebulaai/orion-script-converter).
    
    This convertor will analyze the python code (.py) or jupyter notebook (.ipynb) and ask you a few questions to configure the task. 
    After install the Orion-Script-Converter, start the convertor by typing command "convert2or".
    **Note:**
    - Enter your project
    - Type command 'convert2or'
    
        Input parameters according to the prompt:
        
        1.(Required) Project path: 
	    (Press 'Enter' or '.' for the current directory, '..' for the parent directory of the current folder): 
        
        Input the Python3 project path, either relative path or absolute path. 
        'Enter' or '.' represents the current folder(default) and the '..' means the parent folder 
        of the current path.
        
        2.(Required) entry-point file path(executable file path):
        
        Input the name of entry-point file. This path should inside the Project path.
        
        3.Data configuration: 
	        Do you have external data(data stored outside your project database)
	        that needs to be downloaded from a specific uri (y/n)?
	        
        Set data configuration. If 'y', the following two inputs prompt. Otherwise, this step will skip.
        
            External data uri:  
            
            Input the data uri to get your external data
            
            Path to save the downloaded data within your project:
            
            Input the path(inside your project) to save your downloaded external data.  
        
        For example, in this testing case, you need to download external data from "http://mattmahoney.net/dc/text8.txt" and saved it in the
        the directory of "./datadir/".    
                
        4.Path for the task results(project output directory):
        
            Your project output directory holds your output files. 
            If you have such a directory in your project, input it here. 
            Otherwise, there will be no output files.
            
        5.A NBAI task will be created and saved in the 'task_files' folder 
           which is a sibling folder of your project. 

-Step two: submit your AI zip file to Orion. Please follow [this procedure](https://www.youtube.com/watch?v=FzFNgC4sL3g)


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
