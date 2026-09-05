from email.mime import message
import pickle
from nlp_preprocessing import preprocess_text
#from train import preprocess_text



# Load TF-IDF Vectorizer
with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load Best Model (Random Forest)
with open("../models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)


def predict_message(message):
    
    processed_text = preprocess_text(message)             # Preprocess text
    vectorized_text = tfidf.transform([processed_text])   # Convert text to TF-IDF features
    prediction = model.predict(vectorized_text)[0]        # Prediction
    probability = model.predict_proba(vectorized_text)[0] # Prediction probabilities
    confidence = float(round(max(probability), 4))        # Highest probability as confidence

    if prediction == 1:
        return {
            "prediction": "Spam",
             "confidence": f"{confidence * 100:g}%"
        }

    return {
        "prediction": "Ham",
         "confidence": f"{confidence * 100:g}%"
    } 
    
    # this is short form
    # label = "Spam" if prediction_code == 1 else "Ham"
    # return {"prediction": label, "confidence": confidence}


def inspect_message(message):   #Debug helper to see how preprocessing transforms a message.
   
    processed_text = preprocess_text(message)
    
    print("Original Message: ", message)
    print("Processed Message: ", processed_text)
    
    return {
        "Original Message": message,
        "Processed Message": processed_text
    }