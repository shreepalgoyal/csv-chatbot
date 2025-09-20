import nltk

nltk.download('punkt') #optional if you have already

#make sure to install the pip, pandas, nltk as per requirements

from openai import OpenAI
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

# Creating openai client 
client = OpenAI(api_key='<your OpenAI api key here>')

#for loading files
fee_df = pd.read_csv("fee.csv")
student_df = pd.read_csv("student.csv")

def fee_chatbot(query, prior_answers=None):
    #initailizing gen_txt
    generated_text = ""
    
    #checkout askCsv.py
    
    return {'answer': generated_text, 'query': query}

def student_chatbot(query, prior_answers=None):
    # Initialize generated_text
    generated_text = ""

    #checkout askCsv.py
    
    return {'answer': generated_text, 'query': query}

def compute_bleu_score(chatbot_answers, refrence_answers):
    bleu_scores = []
    for chatbot_answer, refrence_answer in zip(chatbot_answers, refrence_answers):
        chatbot_response = chatbot_answer['answer']
        refrence_response = refrence_answer
        
        #for response to tokens
        chatbot_tokens = nltk.word_tokenize(chatbot_response.lower())
        reference_tokens = nltk.word_tokenize(refrence_response.lower())
        
        #to compute bleu score 
        bleu = sentence_bleu([reference_tokens], chatbot_tokens)
        bleu_scores.append(bleu)
        
    #to compute average bleu score 
    avg_bleu_score = sum(bleu_scores)/len(bleu_scores)
    return avg_bleu_score

# using it as example
visible_sample_questions_fees = [{"query": "What is the duration of the 'B.Sc (Physics, Chemistry, Mathematics, Computer Science, Botany, Zoology)' program?", 
                                  "answer": "The duration of the 'B.Sc (Physics, Chemistry, Mathematics, Computer Science, Botany, Zoology)' program is 3 years."}]

#compleyte bleu score for visible sample questions like: 
bleu_score = compute_bleu_score([fee_chatbot(x['query']) for x in visible_sample_questions_fees], [x['answer'] for x in visible_sample_questions_fees])
print("BLEU Score: ", bleu_score)
