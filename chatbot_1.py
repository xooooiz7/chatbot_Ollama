from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    PostbackTemplateAction
)
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

URI = "neo4j://localhost:7999"
AUTH = ("neo4j", "password")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

def save_user_info(uid, name):
    query = '''
    MERGE (u:User {uid: $uid})
    SET u.name = $name
    '''
    run_query(query, parameters={'uid': uid, 'name': name})

def get_user_name(uid):
    query = '''
    MATCH (u:User {uid: $uid})
    RETURN u.name AS name
    '''
    result = run_query(query, parameters={'uid': uid})
    return result[0]['name'] if result else None

def log_chat_history(uid, message, reply):
    query = '''
    MATCH (u:User {uid: $uid})
    CREATE (c:Chat {message: $message, reply: $reply, timestamp: timestamp()})
    CREATE (u)-[:SENT]->(c)
    '''
    run_query(query, parameters={'uid': uid, 'message': message, 'reply': reply})

def save_response(uid, answer_text, response_msg):
    # Cypher query to match User and create Answer and Response nodes with relationships
    query = '''
    MATCH (u:User {uid: $uid})
    CREATE (a:Answer {text: $answer_text})
    CREATE (r:Response {text: $response_msg})
    CREATE (u)-[:useranswer]->(a)
    CREATE (a)-[:response]->(r)
    '''
    parameters = {
        'uid': uid,
        'answer_text': answer_text,
        'response_msg': response_msg
    }
    run_query(query, parameters)

def compute_response(sentence):
    greeting_corpus = list(set(record['name'] for record in run_query('MATCH (n:Greeting) RETURN n.name as name;')))
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    greeting_scores = util.cos_sim(greeting_vec, ask_vec)
    
    max_greeting_score_index = np.argmax(greeting_scores.cpu().numpy())
    if greeting_scores[max_greeting_score_index] > 0.8:
        match_greeting = greeting_corpus[max_greeting_score_index]
        my_cypher = f"MATCH (n:Greeting) WHERE n.name = '{match_greeting}' RETURN n.msg_reply AS reply"
        results = run_query(my_cypher)
        return results[0]['reply'] if results else None
    return None

def check_previous_question(question):
    cypher_query = '''
    MATCH (q:Question {text: $question})-[:HAS_ANSWER]->(a:Answer)
    RETURN a.text AS answer
    '''
    result = run_query(cypher_query, parameters={"question": question})
    return result[0]['answer'] if result else None

def is_similar_query(user_query, expected_queries):
    user_vec = model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    for expected in expected_queries:
        expected_vec = model.encode(expected, convert_to_tensor=True, normalize_embeddings=True)
        score = util.cos_sim(user_vec, expected_vec)
        if score > 0.7:  # Adjust threshold as needed
            return True
    return False

def remove_endings(text):
    endings = ["ครับ", "ค่ะ", "น้ะ", "นะ", "นะจ้ะ"]
    for ending in endings:
        text = text.replace(ending, "")
    return text.strip()

app = Flask(__name__)

with open('/Users/sittasahathum/Desktop/social/venv/username_line.txt', 'r') as file:
    lines = file.readlines()
    channel_access_token = lines[0].strip()  
    channel_secret = lines[1].strip()          

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        line_bot_api = LineBotApi(channel_access_token)
        handler = WebhookHandler(channel_secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        uid = json_data['events'][0]['source']['userId']

        # Remove ending phrases
        msg = remove_endings(msg)
        
        # Check for name input
        if "ชื่อ" in msg and "อะไร" in msg:
            user_name = get_user_name(uid)
            if user_name:
                line_bot_api.reply_message(tk, TextSendMessage(text=f"ชื่อของคุณคือ {user_name} ค่ะ"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขอโทษค่ะ ฉันไม่ทราบชื่อของคุณ"))

        elif "ชื่อ" in msg and "เชื่อ" not in msg:
            name = msg.split("ชื่อ")[-1].strip()
            if name:
                save_user_info(uid, name)
                line_bot_api.reply_message(tk, TextSendMessage(text=f"ขอบคุณที่แนะนำตัวค่ะ {name}"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ไม่สามารถระบุชื่อได้ กรุณาระบุชื่อของคุณค่ะ"))

        # Respond to name inquiries
        user_name = get_user_name(uid)
        if user_name and is_similar_query(msg, ["ชื่ออะไร", "ผมชื่ออะไร", "ชื่อของฉัน"]):
            line_bot_api.reply_message(tk, TextSendMessage(text=f"ชื่อของคุณคือ {user_name} ค่ะ"))

        response_msg = compute_response(msg)

        if response_msg:
            line_bot_api.reply_message(tk, TextSendMessage(text=response_msg + " ค่ะ"))
            log_chat_history(uid, msg, response_msg)  # Log the chat history
        else:
            previous_answer = check_previous_question(msg)
            if previous_answer:
                line_bot_api.reply_message(tk, TextSendMessage(text=previous_answer + " ค่ะ"))
            else:
                payload = {
                    "model": "supachai/llama-3-typhoon-v1.5",
                    "prompt": f"ผู้ถามชื่อ คุณ{user_name} ตอบสั้นๆไม่เกิน 20 คำ เกี่ยวกับ '{msg}'",
                    "stream": False
                }
                response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    decoded_text = response.json().get("response", "")
                    line_bot_api.reply_message(tk, TextSendMessage(text=decoded_text + " ค่ะ\n.....คำตอบจาก Ollama..."))
                    save_response(uid, msg, decoded_text)  # Save the answer and response
                else:
                    print(f"Failed to get a response from Ollama: {response.status_code}, {response.text}")
                    line_bot_api.reply_message(tk, TextSendMessage(text="เกิดข้อผิดพลาดในการติดต่อ LLaMA"))

    except InvalidSignatureError:
        print("Invalid signature.")
    except Exception as e:
        print("Error:", e)
        print(body)
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
