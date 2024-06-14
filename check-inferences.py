from tensorflow.keras.models import load_model
from utils.embeddings_utils import get_embedding
import numpy as np
import spacy
import pandas as pd
from tqdm import tqdm

test_messages = pd.read_csv('test_sample_messages.csv')

model_files = ['final_model_en_no-replace.h5', 'final_model_en_replace.h5', 'final_model_es_no-replace.h5', 'final_model_es_replace.h5']
embedding_model = "text-embedding-3-large"
output = pd.DataFrame(columns=["message", "model", "NE anonymization", "label_description", "classification_value"])

for model_file_name in model_files:
    # Load model
    model = load_model(model_file_name)
    model_lang = "en" if "_en_" in model_file_name else "es"

    # Define nlp and remove_ne from model_file name
    nlp_model = 'en_core_web_lg' if model_lang == "en" else 'es_core_news_lg'
    remove_ne = False if 'no-replace' in model_file_name else True

    # Replace entities and embed message
    def process_message(message, nlp_model, embedding_model, remove_ne=False):
        
        nlp = spacy.load(nlp_model)

        # Function to remove URLs and find named entities using spaCy
        def process_text(text):
            # First clean the text by removing URLs
            clean_text = " ".join([token.text for token in nlp(text) if not token.like_url])
            
            # Then apply entity detection on the cleaned text
            entities = [(ent.text, ent.label_) for ent in nlp(clean_text).ents]
            return clean_text, entities

        # Apply the process_textfunction to the message
        message_processed, entities = process_text(message)

        # Replace entity values in 'message_processed' with placeholders for generic types
        if remove_ne:
            for entity, type_ in entities:
                message_processed = message_processed.replace(entity, f"<{type_}>")

        # Embed message
        message_processed = get_embedding(message_processed, embedding_model)

        return message_processed

    for index, row in test_messages.iterrows():
        test_message, test_lang = row['text'], row['language']
        
        if test_lang == model_lang:
            # Embed message and replace entities if selected
            message_processed = process_message(test_message, nlp_model, embedding_model, remove_ne)

            # Classify message
            def classify_message(embedded_vector):
                prediction = model.predict(np.array([embedded_vector]))
                class_label = np.argmax(prediction)
                max_output = np.max(prediction)
                return class_label, max_output

            classified_label, classification_value = classify_message(message_processed)
            label_description = "CONSPIRACY" if classified_label == 0 else "CRITICAL"
            
            # Append to output
            new_row = {
                "message": test_message,
                "model": "English Model" if "_en_" in model_file_name else "Spanish Model",
                "NE anonymization": "No" if "no-replace" in model_file_name else "Yes",
                "label_description": label_description,
                "classification_value": round(classification_value, 3)
            }
            output = pd.concat([output, pd.DataFrame([new_row])], ignore_index=True)

# Generate output
output = output.sort_values(by='message')

output.to_csv('output.csv', index=False)

output['message_truncated'] = output['message'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
output = output.drop(columns=['message'])
output = output[['message_truncated', 'model', 'NE anonymization', 'label_description', 'classification_value']]
print(output)
