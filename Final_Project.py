import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
import torch
import yaml
import os
from pathlib import Path
import io


df = pd.read_csv(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Final_Project\Final_new.csv")
EDA_df1 = pd.read_csv("D:\Guvi_Data_Science\MDT33\Capstone_Project\Final_Project\EDA_df1.csv")


# --------------------------------------------------Logo & details on top

st.set_page_config(page_title= "Bank Risk Controller System | By Sathya",
                   layout= "wide",
                   initial_sidebar_state= "expanded")

# Function to safely convert to sqrt
def log_trans(value):
    try:
        return np.log(float(value))  # Conversion to float
    except (ValueError, TypeError):
        raise ValueError(f"Invalid input: {value}")

# Define occupation types in alphabetical order with corresponding numeric codeslabel_encoding

occupation = {
    'Accountants' : 0, 'Cleaning staff' : 1, 'Cooking staff' : 2,
    'Core staff' : 3,  'Drivers' : 4,  'HR staff': 5,  'High skill tech staff' : 6, 'IT staff' : 7,
    'Laborers' : 8, 'Low-skill Laborers' : 9,  'Managers' : 10, 'Medicine staff' : 11, 'Private service staff' : 12,
    'Realty agents': 13, 'Sales staff' : 14,     'Secretaries' : 15,     'Security staff' : 16,     
    'Waiters/barmen staff' : 17}


# Mapping for NAME_EDUCATION_TYPE
education = {'Secondary / secondary special' : 4, 
             'Higher education' : 1, 
             'Incomplete higher' : 2, 
             'Lower secondary' : 3, 'Academic degree' : 0}

# Mapping for Gender
Gender = {'M' : 1,'F' : 0, 'XNA' : 2}

Income = {'Working' : 5, 'State servant' : 3, 'Commercial associate' : 0, 'Student' : 4,
       'Pensioner' : 2, 'Maternity leave' : 1}

Reject_reason = {'XAP(X-Application Pending)' : 7, 'LIMIT(Credit Limit Exceeded)' : 2, 'SCO(Scope of Credit)' : 3,
                'HC(High Credit Risk)' : 1, 'VERIF(Verification Failed)' : 6, 'CLIENT(Client Request)' : 0, 
                'SCOFR(Scope of Credit for Rejection)' : 4, 'XNA(Not Applicable)' : 8, 'SYSTEM(System Error)' : 5}

status = {'Approved' : 0.0, 'Canceled' : 1.0, 'Refused' : 2.0, 'Unused offer' : 2.5}

Yield = {'low_normal' :3, 'middle' :4, 'XNA' :0, 'high' :1, 'low_action' :2}


with st.sidebar:
    st.image(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Final_Project\Risk_2.jpg")
   
    opt = option_menu("Menu",
                    ["Home",'Matrix Insights','EDA','Model Prediction','ML Sentiment Analysis', "YOLO Object Detection","Conclusion"],
                    icons=["house","table","bar-chart-line","graph-up-arrow","search", "binoculars", "exclamation-circle"],
                    menu_icon="cast",
                    default_index=0,
                    styles={"icon": {"color": "Yellow", "font-size": "20px"},
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px", "--hover-color": "blue"},
                            "nav-link-selected": {"background-color": "blue"}})
   
if opt=="Home":
    
        col,coll = st.columns([1,4],gap="small")
        with col:
            st.write(" ")
        with coll:
            st.markdown("# BANK RISK CONTROLLER SYSTEM")
          
            st.write(" ")     
        st.markdown("""
        ### :yellow[OVERVIEW]
        The goal of this project is to develop a reliable predictive model 
        that effectively identifies customers at high risk of loan default. 
        This will allow the financial institution to proactively manage its credit portfolio, 
        implement targeted strategies, and ultimately minimize the likelihood of loan defaults.
        """)

        col1,col2=st.columns([3,2],gap="Medium")
        with col1:
            st.markdown("""
                        ### :Yellow[DOMAIN] Banking
                        """)
            st.markdown("""
                        ### :Yellow[TECHNOLOGIES USED]     
                        ##### Python
                        ##### Data Preprocessing
                        ##### EDA(Exploratory Data Analysis)
                        ##### Pandas
                        ##### Numpy
                        ##### Visualization
                        ##### Machine Learning - Classification Model
                        ##### Streamlit GUI
                        
                        """)
        with col2:
                st.write(" ")
    
if opt=="Matrix Insights":
                st.header(":Blue[DataFrame and Matrix Insights]")
                st.dataframe(EDA_df1.head(5))

                st.header(":blue[Model Performance]")
                data = {
                            "Algorithm": ["Decision Tree","KNN","Random Forest","XGradientBoost"],
                            "Accuracy": [95,93,94,93],
                            "Precision": [95,93,95,1],
                            "Recall": [95,93,95,1],
                            "F1 Score": [95,93,95,1]
                            
                            }
                dff = pd.DataFrame(data)
                st.dataframe(dff)
                st.markdown(f"## The Selected Algorithm is :green[*Decision Tree*] and its Accuracy is   :green[*95%*]")

if opt=="EDA":
      
    st.subheader(":orange[Insights of Bank Risk Controller System]")
    st.write(":Yellow[**Only Target=1 data included for Analysis**]")

    col1,col2 = st.columns(2)
    
    def plot_def1(col):
    # Filter data where TARGET is 1 and count the occurrences of the specified column
        data_1 = EDA_df1[EDA_df1['TARGET'] == 1][col].value_counts()
        data_1_df = data_1.reset_index()
        data_1_df.columns = [col, 'Count']
    
    # Display the top 10 values in the Streamlit app
        color_discrete_map = {'M': 'blue', 'F': 'pink'}
    
    # Create the bar plot using Plotly
        fig = px.bar(data_1_df, 
                 x=col, 
                 y='Count', 
                 labels={ 'x': col, 'y': 'Count' }, 
                 color=col, 
                 color_discrete_map=color_discrete_map,
                 title=f"{col}",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Display the plot in the Streamlit app
        st.plotly_chart(fig)
    
    def plot_def2(col):
    # Filter data where TARGET is 1 and count the occurrences of the specified column
        data_1 = EDA_df1[EDA_df1['TARGET'] == 1][col].value_counts()
        data_1_df = data_1.reset_index()
        data_1_df.columns = [col, 'Count']
    
    # Display the top 10 values in the Streamlit app
        color_discrete_map = {'M': 'blue', 'F': 'pink'}
    
    # Create the bar plot using Plotly
        pie_fig = px.pie(data_1_df, 
                     names=col, 
                     values='Count', 
                     color=col, 
                     color_discrete_map=color_discrete_map,
                     title=f"{col} - Pie Chart",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Display the plot in the Streamlit app
        st.plotly_chart(pie_fig)

    def plot_numerical_vs_target(data, num_col, target_col):
    # Filter the data where the target column is equal to 1
        
        filtered_data = data[data[target_col] == 1]
                
    # Create the histogram using Plotly
        fig = px.histogram(filtered_data, 
                       x=num_col, 
                       nbins=30,  # You can adjust the number of bins if necessary
                       title=f'{num_col} Distribution',
                       color_discrete_sequence=['orange'])
    
        # Update layout to better format the plot
        fig.update_layout(
        xaxis_title=num_col,
        yaxis_title='Count',
        bargap=0.2  # Adjust the gap between bars
         )
    
        # Show the plot in Streamlit
        st.plotly_chart(fig)

    with col1:
        plot_def1("CODE_GENDER")
        plot_def1("OCCUPATION_TYPE")
        plot_def1("CODE_REJECT_REASON")
    
    with col2:
        plot_def1("NAME_EDUCATION_TYPE")
        plot_def2("NAME_GOODS_CATEGORY")
        plot_def1("PRODUCT_COMBINATION")

    plot_numerical_vs_target(EDA_df1, "AGE","TARGET")


if opt=="Model Prediction":
            
    # Streamlit form for user inputs
    st.markdown(f'## :blue[Predicting Customers Default on Loans]')
    st.write(" ")
    
    with st.form("my_form"):
        col1, col2 = st.columns([5, 5])
               
        with col1:

            OCCUPATION_TYPE = st.selectbox("OCCUPATION TYPE", list(occupation.keys()), key='OCCUPATION_TYPE')
            EDUCATION_TYPE = st.selectbox("EDUCATION TYPE", list(education.keys()), key='EDUCATION_TYPE')
            NAME_INCOME_TYPE = st.selectbox("INCOME TYPE",list(Income.keys()), key='NAME_INCOME_TYPE')
            TOTAL_INCOME = st.number_input("TOTAL INCOME PA", key='TOTAL_INCOME', format="%.2f")
            CODE_REJECT_REASON = st.selectbox("CODE REJECTION REASON",list(Reject_reason.keys()), key='CODE_REJECT_REASON')
            NAME_CONTRACT_STATUS = st.selectbox("CONTRACT STATUS",list(status.keys()), key='NAME_CONTRACT_STATUS')
            NAME_YIELD_GROUP = st.selectbox("YIELD GROUP",list(Yield.keys()), key='NAME_YIELD_GROUP')

        with col2:
                    
            CODE_GENDER = st.selectbox("CODE GENDER", list(Gender.keys()), key='CODE_GENDER')
            AGE = st.text_input("AGE", key="AGE")
            CLIENT_RATING = st.text_input("CLIENT RATING", key="CLIENT_RATING")
            DAYS_LAST_PHONE_CHANGE = st.number_input("PHONE CHANGE", key="DAYS_LAST_PHONE_CHANGE", format="%.2f")
            DAYS_ID_PUBLISH = st.number_input("DAYS ID PUBLISH", key="DAYS_ID_PUBLISH", format="%.2f")
            DAYS_REGISTRATION = st.number_input("DAYS REGISTRATION", key="DAYS_REGISTRATION", format="%.2f")
            DAYS_EMPLOYED_log = st.number_input("DAYS EMPLOYED", key='DAYS_EMPLOYED_log', format="%.5f")
           
        submit_button = st.form_submit_button(label="PREDICT STATUS")

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #ADD8E6;
            color: green;
            width: 50%;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    flag = 0
    if submit_button:
        try:
            for i in [TOTAL_INCOME, DAYS_EMPLOYED_log, AGE, CLIENT_RATING, DAYS_LAST_PHONE_CHANGE, DAYS_ID_PUBLISH, DAYS_REGISTRATION]:             
                if i is None or i == '':
                    flag = 1
                    break
        except ValueError:
            flag = 1
    if flag == 1:
        st.write("Please enter a valid number. Fields cannot be empty.")
    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("Please enter a valid number, space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)  
    
    if submit_button and flag == 0:
        
        try:
            # Encode categorical variables
            le = LabelEncoder()

            Occupation = occupation[OCCUPATION_TYPE]
            Education = education[EDUCATION_TYPE]
            Income_type = Income[NAME_INCOME_TYPE]
            Income_amt = int(TOTAL_INCOME)
            Days_employed = int(DAYS_EMPLOYED_log)  
            Reason = Reject_reason[CODE_REJECT_REASON]
            Status = status[NAME_CONTRACT_STATUS]
            yield_group = Yield[NAME_YIELD_GROUP]
            Genders = Gender[CODE_GENDER]
            Age  = int(AGE.strip())
            Rating = int(CLIENT_RATING.strip())
            Phone  = int(DAYS_LAST_PHONE_CHANGE)
            ID_Published = int(DAYS_ID_PUBLISH)
            Registration = int(DAYS_REGISTRATION)

            # Create sample array with encoded categorical variables
            sample = np.array([
                [
                    Occupation,
                    Education,
                    Income_type,
                    Income_amt,
                    Reason,
                    Status,
                    yield_group,
                    Genders,
                    Age,
                    Rating,
                    Phone, 
                    ID_Published, 
                    Registration,
                    log_trans(Days_employed) 
                ]
            ])
            Dir = "D:/Guvi_Data_Science/MDT33/Capstone_Project/Final_Project/"    
            with open(Dir + "dtmodel.pkl", 'rb') as file:
                Decision_tree = pickle.load(file)

            #sample = scaler_loaded.transform(sample)
            pred = Decision_tree.predict(sample)

            if pred == 1:
                st.markdown(f' ## The status is: :red[Won\'t Repay]')
            else:
                st.write(f' ## The status is: :green[Repay]')
        except ValueError as e:
            st.error(f"Error processing inputs: {e}")
            st.write("Please check your input values. Only numeric values are allowed.")

if opt == "ML Sentiment Analysis":

    st.markdown("### :Green[ML Sentiment Analysis]")
    st.write("")
    st.write("")

    # Initialize the sentiment analyzer
    nltk.download('vader_lexicon') #VADER (Valence Aware Dictionary and sEntiment Reasoner)
    sia = SentimentIntensityAnalyzer()

    # Create a function to analyze the sentiment
    def analyze_sentiment(text):
        sentiment = sia.polarity_scores(text)
        if sentiment['compound'] >= 0.05:
            return "Positive"
        elif sentiment['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Create a Streamlit app
    st.title("Sentiment Analysis App")

    # Get the user input
    text = st.text_input("Enter some text:")

    # Check if text is not empty
    if text:
        # Analyze the sentiment
        sentiment = analyze_sentiment(text)

        # Display the sentiment as a word
        st.write("Sentiment:", sentiment)

        # Get the sentiment scores
        sentiment = sia.polarity_scores(text)

        # Display the bar chart
        st.bar_chart({'Positive': sentiment['pos'], 'Negative': sentiment['neg'], 'Neutral': sentiment['neu']})

if opt=="YOLO Object Detection":
     
     YOLO_DIR = r"C:\Users\Admin\yolov5"
     WEIGHTS_PATH = os.path.join(YOLO_DIR, "yolov5s.pt")

     os.chdir(YOLO_DIR)
     model = torch.hub.load(YOLO_DIR, "custom", path=WEIGHTS_PATH, source='local')

     st.title("YOLO Object Detection with Custom Trained Model")
     st.write("Upload an image and click submit to detect objects using the trained YOLO model.")

     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

     if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button('Submit and Detect'):
            st.write("Detecting objects...")
            
            #Perform detection
            results = model(img)  # Pass the PIL image directly to the model
            results.save()  # Save results to `runs/detect/exp` directory
            
            # Dynamically find the latest `exp` directory
            exp_dirs = sorted(Path("runs/detect").glob("exp*"), key=os.path.getmtime, reverse=True)
            
            if exp_dirs:
                exp_dir = exp_dirs[0]  # Get the latest directory
                #st.write(f"Most recent experiment directory: {exp_dir}")
                for file in os.listdir(exp_dir):
                    st.write(f"File found: {file}")
                
                detected_img_path = None
                for file in os.listdir(exp_dir):
                    if file.endswith(".jpg") or file.endswith(".png"):
                        detected_img_path = os.path.join(exp_dir, file)
                        break
                
                if detected_img_path and os.path.exists(detected_img_path):
                    detected_img = Image.open(detected_img_path)
                    st.image(detected_img, caption="Detected Image", use_column_width=True)
                    st.write("Detection Complete.")
                else:
                    st.error("Detected image not found in the expected location.")
            else:
                st.error("Detection results folder not found.")

if opt=="Conclusion":
               
        st.markdown(f"### :green[Conclusion]")
        st.markdown(f"#### In the financial industry, Default occurs when a borrower fails to meet the legal obligations of a loan. The ML Model Streamlit App can accurately identify the customers who are likely to default on their loans based on their historical data.")
        st.markdown(f"#### It also provides deep insights on the features which can helps to predict the customars who are likely to default on their loans")
        st.markdown(f"#### This will enable the financial institution to proactively manage their credit portfolio and ultimately reduce the risk of loan defaults")
        st.write(" ")
        st.markdown("https://github.com/SathyaMadhu/Bank-Risk-Controller-System")    





            


    








