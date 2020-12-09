# THACC

## Initials
After git pull either via terminal or GUI do this

```
cd THACC_Pytorch
pip install --upgrade pip
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Folder Structure
```
THACC/
    - data
        - new_data
            - new email file in excel format
        - train_data
            - type/
            - message/
        - test_data
            - user email file in excel format
    - training
        - data_prep.py
        - financial_emails_training.py
        - utils_train.py
    - weights
        - level1
            - model.pt
            - vocab.pkl
        - level2
            - meta.json
            - tokenizer
            - ner/
            - vocab/
    - plots
        -plots.png
    - api
        - app.py
        - timeline_prediction.py
        - utils_pred.py
    - requirements.txt
    - Procfile
    - runtime.txt
    - README.md
    - .gitignore
```

## Tagging

```
- Once you get a file in 'new_data' folder Tag emails as 'Finance', 'MaybeUseful', 'NotFinance' by adding a column 'Type' and save it as excel file inside 'type' folder in 'train' folder and delete the file in 'new_data'.
- python training/data_prep.py --input data/train_data/type --output_txt data/train_data/message
- Label the files from 'message' folder and annotate the text with 25 provided encoding labels and save the output of doccano as json file inside 'message' folder
```

## Training
```
- python training/financial_emails_training.py --input_type data/train_data/type/ --input_message data/train_data/message/ --model_type weights/level1/model.pt --vocab weights/level1/vocab.pkl --model_ner weights/level2/ --plot plots/plot1.png
```

## Run API for prediction on user data:
```
- python api/app.py
```
