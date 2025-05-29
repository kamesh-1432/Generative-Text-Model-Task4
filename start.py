# Step 2: Install ngrok and run Streamlit
!pip install pyngrok
from pyngrok import ngrok

# Replace with your ngrok auth token (get from https://ngrok.com)
!ngrok authtoken 2xiQzDPZmKAFrrFoUxMNqPKUopu_2Aw42P87UoDifyobRP1sv

# Start Streamlit app
!streamlit run /content/drive/MyDrive/Generative-Text-Model/app.py &>/dev/null &

# Create public URL
public_url = ngrok.connect(8501)
print(f"Streamlit app running at: {public_url}")