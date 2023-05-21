## FalseQA
This repository contains the code and dataset for the ACL 2023 paper titled Won't Get Fooled Again: Answering Questions with False Premises.

## Dataset
The main content of this repository is the dataset proposed in our work, which can be found in the data directory. Each CSV file consists of three columns: question, answer, label.

question: This column contains the questions, including both false premise questions and true premise questions.
answer: This column contains the corresponding answers. It includes normal answers for true premise questions and rebuttals for false premise questions.
label: This column contains the binary label for each question. The value 1 indicates a false premise question, while 0 indicates a true premise question.

## Scripts
To conduct the few-shot experiments, we randomly sample from the train.csv file and store the samples in a temporary file named train_{scale}shots_{seed}seed.csv. We then utilize the scripts located in the scripts/ directory to run the experiments.

## Update
This work was completed prior to the rise of ChatGPT. However, we have tested ChatGPT's performance on our dataset and found it to be remarkably good. This may be attributed to the fact that the instruction tuning data contains false premise questions, as highlighted in the InstructGPT and GPT-4 reports. Currently, we are working on adapting our dataset to the instruction tuning format.