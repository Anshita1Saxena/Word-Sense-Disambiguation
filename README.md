# Word-Sense-Disambiguation
This repository is created for showing the Lesk and Self-Supervised Yarowsky Implementation to perform the word sense disambiguation task.

## Code Information:-
loader_main.py is the main code from where all the experiments will run. 
Four model files are `most_frequent_model.py`, `lesk_model.py`, `custom_model.py`, and `bootstrapping_model.py` 
Utility functions are defined in `utils.py` file.
Files like pull_sentences.py and sentence_sense_data.py were used for the creation and extraction of the dataset.

## Dataset Information:-
`final_deal.csv`, `final_level.csv`, `final_power.csv`, `final_sense.csv`, `final_time.csv` are the dataset files for running the bootstrapping method.

## Experiment File Information:-
`results_log_file.txt` contains the accuracy of all the model experiments, which is generated once we run the code. It contains 78 experiments done on 4 models with different parameter settings.

To run this code-script, the command is given below:-

python -W ignore .\loader_main.py

## Collective Information
`nlp_report_final.pdf` file contains the information related to problem setup, dataset generation, analysis and model discussion, results, and model success/failures for improvement. 

## Feedback Info
![feedback text](https://github.com/Anshita1Saxena/Word-Sense-Disambiguation/blob/main/images/img.png)
