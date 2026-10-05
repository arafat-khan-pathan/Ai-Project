# Spam Detection Project

A machine learning project for classifying messages as spam or ham.

## Setup and Run

Run these commands from the project root:

```powershell
pip install -r requirements.txt
```
OR

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train the model if needed:

```powershell
python train.py
```

Start the local server:

```powershell
python server.py
```

Open `http://localhost:8000` in a browser.

## File Structure

```text
.
│   .gitignore
│   README.md
│   requirements.txt
│   server.py
│   train.py
│
├───dataset
│       new_split_.ipynb
│       sep_ham.csv
│       sep_spam.csv
│       spam.csv
│       test-ham-spam.csv
│       train.csv
│
├───models
│       random_forest.pkl
│       tfidf_vectorizer.pkl
│
├───notebooks
│       final_demo.ipynb
│       score_find.ipynb
│
├───src
│       nlp_preprocessing.py
│       prediction.py
│
└───templates
        intex.html
```

## Acknowledgments
This project is based on the reference repository by [Saiful-alam105](https://github.com/Saiful-alam105/spam-detection.git)
