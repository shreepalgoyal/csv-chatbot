import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv

#make sure to install the pip, streamlit, langchain & langchain_experiments, dotenv as per requirements

#run it by - Terminal > streamlit run askCsv.py

def main():
    
    #loads the openai key client
    load_dotenv()
    
    #making it easy to run by GUI 
    
    
    st.set_page_config(page_title="Ask you CSV 🗃️")
    st.header("Ask you CSV 🗃️")
    
    user_csv = st.file_uploader("Upload your CSV file", type="csv")
    
    if user_csv is not None:
        user_question = st.text_input("Ask a question for your CSV: ")
        
        llm = OpenAI(temperature=0)
        agent = create_csv_agent(llm, user_csv, verbose=True)
        
        if user_question is not None and user_question != "":
            
            response = agent.run(user_question)
            
            st.write(response)
        
    

if __name__ == "__main__":
    main()