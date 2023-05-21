import streamlit as st
import pandas as pd
from skllm.config import SKLLMConfig
from skllm import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset
import os 

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ORGANIZATION_NAME = os.getenv('ORGANIZATION_NAME')

SKLLMConfig.set_openai_org(ORGANIZATION_NAME)
SKLLMConfig.set_openai_key(OPENAI_API_KEY)

X, y = get_classification_dataset() 
clf = ZeroShotGPTClassifier(openai_model = "gpt-3.5-turbo")

clf.fit(X, y)

st.set_page_config(layout="wide")

upload_csv = st.file_uploader("Upload your CSV file", type=['csv'])
if upload_csv is not None:
    if st.button("Analyze CSV File"):
        col1, col2 = st.columns([1,1])
        with col1:
            st.info("CSV File uploaded")
            csv_file = upload_csv.name
            with open(os.path.join(csv_file),"wb") as f: 
                f.write(upload_csv.getbuffer()) 
            print(csv_file)
            df = pd.read_csv(csv_file, encoding= 'unicode_escape', index_col=None)
            st.dataframe(df, use_container_width=True)
        with col2:
            data_list = df['Review'].tolist()
            labels = clf.predict(data_list)
            df['Sentiment'] = labels
            st.info("Sentiment Analysis Result")
            st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download data as CSV", data=csv_data, file_name='result_df.csv', mime='text/csv')