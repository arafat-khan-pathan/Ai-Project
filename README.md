# Spam Detection Project

A machine learning project for classifying messages as spam or ham.

## Description

This project is a web application that uses a machine learning model to classify text messages as spam or ham. The application uses a trained random forest classifier to make predictions based on the input text. The application also provides a dashboard that displays the accuracy of the model and the number of messages classified as spam and ham.

## Features

* Web-based interface for inputting text and viewing predictions
* Dashboard for displaying model accuracy and classification counts
* Uses a trained random forest classifier for spam detection
* Supports both training and prediction
* Uses a pre-trained TF-IDF vectorizer for text preprocessing
* Uses joblib for model serialization and deserialization

## Installation

To install the project, follow these steps:

1. Clone the repository:
```powershell
git clone https://github.com/arafat-khan-pathan/Spam-Detection.git
```

2. Navigate to the project directory:
```powershell
cd Spam-Detection
```

3. Install the required dependencies:
```powershell
pip install -r requirements.txt
```

4. Start the server:
```powershell
python server.py
```


## Usage

1. Run the `server.py` script to start the web server.
2. Open a web browser and navigate to `http://localhost:8000`.
3. Enter a text message in the input field and click the "Predict" button.
4. The application will use the trained model to classify the message as spam or ham.
5. The result will be displayed on the screen.
6. The dashboard can be accessed by clicking the "Dashboard" button.





## File Structure

```text
Spam-Detection/
|
│   .gitignore
│   README.md
│   requirements.txt
│   server.py
│   train.py
│
├───dataset/
│       new_split_.ipynb
│       sep_ham.csv
│       sep_spam.csv
│       spam.csv
│       test-ham-spam.csv
│       train.csv
│
├───models/
│       random_forest.pkl
│       tfidf_vectorizer.pkl
│
├───notebooks/
│       final_demo.ipynb
│       score_find.ipynb
│
├───src/
│       nlp_preprocessing.py
│       prediction.py
│
└───templates/
        intex.html
```

## Acknowledgments
This project is based on the reference repository by [Saiful-alam105](https://github.com/Saiful-alam105/spam-detection.git)
