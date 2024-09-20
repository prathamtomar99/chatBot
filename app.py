from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

#llm1
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose = True,
                           temprature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))
convo= ConversationChain(llm=llm)

#llm 2
client = InferenceClient(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    token=os.getenv("HUGGINGFACE_API_KEY")
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # try:
    #     # Make a text request to the Hugging Face model
    #     # response = client.text(user_input)
    #     response=""
    #     for message in client.chat_completion(
    #         messages=[{"role": "user", "content": f"{user_input}"}],
    #         max_tokens=10000,
    #         stream=True,
    #     ):
    #         print(message.choices[0].delta.content, end="")
    #         response= response+message.choices[0].delta.content
    #     print(response)
    #     response=response.replace('\n', '<br>')
    #     response=response.split("**")
    #     final_text=""
    #     for i, part in enumerate(response):
    #         final_text +=  f"<span style='font-size: 20.5px;'>{part}</span>" if((i%2 != 0)  & (i>0)) else  part
    #     # print(final_text)
    #     return jsonify({'response': final_text})
    try:
        response=""
        reply=convo.invoke(user_input)
        response=reply['response'].split("AI:")[-1]
        response=response.replace('\n', '<br>')
        response=response.split("**")
        final_text=""
        for i, part in enumerate(response):
            final_text +=  f"<span style='font-size: 20.5px;'>{part}</span>" if((i%2 != 0)  & (i>0)) else  part
        # print(final_text)
        return jsonify({'response': final_text})
    except Exception as e:
        print(e)
        # Handle possible errors
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
