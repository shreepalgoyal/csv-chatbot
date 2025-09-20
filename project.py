import nltk
#nltk.download('punkt')  # Download necessary resources for NLTK
import openai
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Creating OpenAI client 
load_dotenv()

# Load datasets
fee_df = pd.read_csv("fee.csv")
student_df = pd.read_csv("student.csv")

st.set_page_config(page_title="Ask you CSV 🗃️")
st.header("Ask you CSV 🗃️")

def fee_chatbot(query, prior_answers=None):
    
    if fee_df is not None:
        user_question = query
        
        llm = OpenAI(temperature=0.5)
        agent = create_csv_agent(llm, fee_df, verbose=True)
        
        if user_question is not None and user_question != "":
            
            response = agent.run(user_question)
            
            st.write(response)
        
        generated_text = response
    
      # Logic for response will be here 
    return {'answer': generated_text, 'query': query}

def compute_bleu_score(chatbot_answers, reference_answers):
    bleu_scores = []
    for chatbot_answer, reference_answer in zip(chatbot_answers, reference_answers):
        chatbot_response = chatbot_answer['answer']
        reference_response = reference_answer
        
        # Response to tokens
        chatbot_tokens = nltk.word_tokenize(chatbot_response.lower())
        reference_tokens = nltk.word_tokenize(reference_response.lower())
        
        # Compute BLEU score 
        bleu = sentence_bleu([reference_tokens], chatbot_tokens)
        bleu_scores.append(bleu)
        
    # Compute average BLEU score 
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score

# Example usage
visible_sample_questions_fees = [{"query": "What is the duration of the 'B.Sc (Physics, Chemistry, Mathematics, Computer Science, Botany, Zoology)' program?", 
                                  "answer": "The duration of the 'B.Sc (Physics, Chemistry, Mathematics, Computer Science, Botany, Zoology)' program is 3 years."}]

# Compute BLEU score for visible sample questions
bleu_score = compute_bleu_score([fee_chatbot(x['query']) for x in visible_sample_questions_fees], [x['answer'] for x in visible_sample_questions_fees])
st.write("BLEU Score: ", bleu_score)
