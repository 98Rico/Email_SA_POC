import os
import pandas as pd
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import streamlit as st
import json

import matplotlib.pyplot as plt
import seaborn as sns

import ollama

from dotenv import load_dotenv


# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Email_Project_v1",
    page_icon="🔹"
)

#load_dotenv("secrets.env")


    
# Access Gmail token and credentials
token_json = st.secrets["gmail"]["token_json"]
credentials_json = st.secrets["gmail"]["credentials_json"]

# Access OpenAI API key
openai_key = st.secrets["openai"]["api_key"]

# Access Hugging Face token
hf_token = st.secrets["huggingface"]["token"]

# Access default email
default_email = st.secrets["default"]["email"]

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def authenticate_gmail_old():
    """Authenticate and create the Gmail API service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def authenticate_gmail():
    """Authenticate and create the Gmail API service using Streamlit secrets."""
    creds = None

    # Load JSONs from Streamlit secrets
    token_json = json.loads(st.secrets["gmail"]["token_json"])
    credentials_json = json.loads(st.secrets["gmail"]["credentials_json"])

    # Create credentials from the token JSON
    creds = Credentials.from_authorized_user_info(token_json, SCOPES)

    # Refresh or create new credentials if necessary
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use client configuration for creating new credentials
            flow = InstalledAppFlow.from_client_config(credentials_json["installed"], SCOPES)
            creds = flow.run_local_server(port=0)

    return build('gmail', 'v1', credentials=creds)

def get_emails(service, user_id='me'):
    """Fetches emails from the user's inbox."""
    emails = []
    results = service.users().messages().list(userId=user_id).execute()
    messages = results.get('messages', [])

    for message in messages[:10]:  # Fetching only the first 10 emails
        msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
        payload = msg['payload']
        headers = payload['headers']

        email_data = {}
        for header in headers:
            if header['name'] == 'From':
                email_data['From'] = header['value']
            if header['name'] == 'Subject':
                email_data['Subject'] = header['value']
            if header['name'] == 'Date':
                email_data['Date'] = header['value']

        # Decode email body
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    email_data['Body'] = base64.urlsafe_b64decode(data).decode('utf-8')
        else:
            data = payload['body']['data']
            email_data['Body'] = base64.urlsafe_b64decode(data).decode('utf-8')

        emails.append(email_data)

    return pd.DataFrame(emails)




def generate_response(email_content):
    input_text = f"Generate a very short (2 sentences max) polite response to the following email: {email_content}"

    # Call the Ollama API to generate a response
    result = ollama.generate(
        model="mistral:latest",  
        prompt=input_text
    )

    # Extract and return the generated response
    return result['response']  

def generate_importance_score(email_content):
    input_text = (
        f"Score the importance of the following email from 0 (spam) to 10 (urgent). "
        "I only want integers, no float. A spam is an auto-generated email, whereas a very important email "
        "is an email from a client that needs something. Only give as an output the number, nothing more: "
        f"{email_content}"
    )

    try:
        # Call the Ollama API to generate a score
        result = ollama.generate(
            model="mistral:latest",  
            prompt=input_text
        )

        # Extract and return the importance score, ensuring it's an integer
        return int(result['response'].strip())  # Use strip() to remove extra whitespace

    except ValueError:
        # Handle case where conversion to int fails
        print("Error: Could not convert response to integer.")
        return None  # Or another appropriate default value

    except Exception as e:
        # Handle other potential exceptions
        print(f"Error during API call: {e}")
        return None  # Or another appropriate default value


def main():

    
    st.title("Email Reply")

    service = authenticate_gmail()
    email_df = get_emails(service)
    
    if 'importance_scores' not in st.session_state:
        with st.spinner("Loading Emails..."):
            # Generate scores using apply() and store in session state
            st.session_state.importance_scores = email_df['Body'].apply(generate_importance_score).tolist()
        
        # Assign scores to the DataFrame
        email_df['Importance'] = st.session_state.importance_scores
    else:
        # Use previously calculated scores from session state
        email_df['Importance'] = st.session_state.importance_scores

    st.dataframe(email_df)


    # Plotting the distribution of importance scores
    st.subheader("Distribution of Importance Scores")

    # Set Seaborn style and create a new figure
    sns.set_theme(style="whitegrid")  # Use a clean and professional theme
    fig, ax = plt.subplots(figsize=(3, 2))  # Define figure size

    # Count the number of emails per importance score
    importance_counts = email_df['Importance'].value_counts().sort_index()

    # Create the bar plot using Seaborn
    sns.barplot(
        x=importance_counts.index,
        y=importance_counts.values,
        #palette="coolwarm",  # Use a vibrant color palette
        ax=ax  # Use the Axes object for better customization
    )

    # Customize the plot's labels, title, and ticks
    ax.set_xlabel('Importance Score', fontsize=8, labelpad=5)
    ax.set_ylabel('Number of Emails', fontsize=8, labelpad=5)
    ax.set_title('Distribution of Email Importance Scores', fontsize=7, pad=7)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    # Remove unnecessary spines for a cleaner look
    sns.despine(left=True, bottom=False)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Filtering emails by importance
    importance_filter = st.slider("Select Importance Range", 0, 10, (0, 10))
    filtered_emails = email_df[(email_df['Importance'] >= importance_filter[0]) & 
                                (email_df['Importance'] <= importance_filter[1])]
    
    st.subheader("Filtered Emails")
    st.dataframe(filtered_emails)
    #####

    st.subheader("Emails")
    selected_email = st.selectbox("Select an email body:", options=filtered_emails['Body'].tolist())

    # Store the selected email body in EmailContent
    if selected_email:
        EmailContent = selected_email
        st.text_area("Email Content", value=EmailContent, height=150)
    
    if st.button("Generate Response"):
        
        if EmailContent:
            with st.spinner("Thinking..."):
                response_content = generate_response(EmailContent)
            st.success("Done!")
            st.write(response_content)
            st.text_area("Generated Response", value=response_content, height=150)

if __name__ == '__main__':
    main()