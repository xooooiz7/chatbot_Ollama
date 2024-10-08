from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import chromedriver_autoinstaller

# Setup Chrome options and install chromedriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chromedriver_autoinstaller.install()

# Constants
OLLAMA_API_URL = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
URI = "neo4j://localhost:7999"
AUTH = ("neo4j", "password")

# Database query functions
def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            return [record for record in session.run(query, parameters)]

def save_user_info(uid, name):
    run_query('MERGE (u:User {uid: $uid}) SET u.name = $name', {'uid': uid, 'name': name})


def get_user_name(uid):
    query = '''
    MATCH (u:User {uid: $uid})
    RETURN u.name AS name
    '''
    result = run_query(query, parameters={'uid': uid})
    return result[0]['name'] if result else None

def log_chat_history(uid, message, reply):
    run_query('MATCH (u:User {uid: $uid}) CREATE (c:Chat {message: $message, reply: $reply, timestamp: timestamp()}) CREATE (u)-[:SENT]->(c)', {'uid': uid, 'message': message, 'reply': reply})

def save_response(uid, answer_text, response_msg):
    run_query('MATCH (u:User {uid: $uid}) CREATE (a:Answer {text: $answer_text}) CREATE (r:Response {text: $response_msg}) CREATE (u)-[:useranswer]->(a) CREATE (a)-[:response]->(r)', {'uid': uid, 'answer_text': answer_text, 'response_msg': response_msg})
def clean_price(price_str):
    # ใช้ regex เพื่อลบจุลภาค จุดทศนิยม และสกุลเงิน
    cleaned_price = re.sub(r'[^\d]', '', price_str)
    return int(cleaned_price)

# Response computation functions
def compute_response(sentence):
    greeting_corpus = list(set(record['name'] for record in run_query('MATCH (n:Greeting) RETURN n.name as name;')))
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    greeting_scores = util.cos_sim(greeting_vec, ask_vec)
    
    if (max_index := np.argmax(greeting_scores.cpu().numpy())) and greeting_scores[max_index] > 0.6:
        match_greeting = greeting_corpus[max_index]
        results = run_query(f"MATCH (n:Greeting) WHERE n.name = '{match_greeting}' RETURN n.msg_reply AS reply")
        return results[0]['reply'] if results else None
    return None

def check_previous_question(question):
    result = run_query('MATCH (q:Question {text: $question})-[:HAS_ANSWER]->(a:Answer) RETURN a.text AS answer', {"question": question})
    return result[0]['answer'] if result else None

def is_similar_query(user_query, expected_queries):
    user_vec = model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    return any(util.cos_sim(user_vec, model.encode(expected, convert_to_tensor=True, normalize_embeddings=True)) > 0.7 for expected in expected_queries)

def remove_endings(text):
    endings = ["ครับ", "ค่ะ", "น้ะ", "นะ", "นะจ้ะ"]
    for ending in endings:
        text = text.replace(ending, "")
    return text.strip()

# Fetch product info
def fetch_product_info(search_term):
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://www.fpvthai.com/search?q={search_term}")
    driver.implicitly_wait(10)
    html = driver.page_source
    driver.quit()

    results = []
    mysoup = BeautifulSoup(html, "html.parser")
    
    # Loop through all product titles
    for title_element in mysoup.find_all("div", class_="list-view-item__title"):
        title_text = title_element.get_text().strip()
        
        # Get the price
        price_element = title_element.find_next("span", class_="price-item price-item--regular")
        price_text = price_element.get_text().strip() if price_element else "Price not available"
        
        # Get the link
        link_element = title_element.find_previous("a", class_="full-width-link")
        link = f"https://www.fpvthai.com{link_element['href']}" if link_element else "Link not available"

        # Append product info
        results.append({
            'title': title_text,
            'price': price_text,
            'link': link
        })
    
    # If no results found
    if not results:
        return None

    return results


# Flask app
app = Flask(__name__)
with open('/Users/sittasahathum/Desktop/social/venv/username_line.txt', 'r') as file:
    channel_access_token, channel_secret = [line.strip() for line in file.readlines()]

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        line_bot_api = LineBotApi(channel_access_token)
        handler = WebhookHandler(channel_secret)
        handler.handle(body, request.headers['X-Line-Signature'])

        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        uid = json_data['events'][0]['source']['userId']
        global search_term 
        global price_min
        msg = remove_endings(msg)
        global is_lower_selected
        if "ตัวเลือก" in msg:
            # Display quick reply options
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label="แบตเตอรี่", text="ค้นหา แบตเตอรี่")),
                QuickReplyButton(action=MessageAction(label="โดรน", text="ค้นหา โดรน")),
                QuickReplyButton(action=MessageAction(label="Accessories", text="ค้นหา Accessories")),
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="กรุณาเลือกประเภทสินค้า:", quick_reply=quick_reply))

        elif "สอบถามข้อมูล" in msg:
            # Display quick reply options
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label="แบตเตอรี่", text="รายละเอียดเพิ่มเติม Battery")),
                QuickReplyButton(action=MessageAction(label="โดรน", text="รายละเอียดเพิ่มเติม Drone")),
                QuickReplyButton(action=MessageAction(label="Accessories", text="รายละเอียดเพิ่มเติม Accessories")),
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="กรุณาเลือกประเภทสินค้า:", quick_reply=quick_reply))

        elif "รายละเอียดเพิ่มเติม Battery" in msg:
            # Display quick reply options for batteries
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label=f"แบตเตอรี่{i}", text=f"รายละเอียดเพิ่มเติม แบตเตอรี่{i}")) for i in ["1s", "2s", "3s", "4s", "6s"]
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="กรุณาเลือกเพิ่มเติม:", quick_reply=quick_reply))

        elif "รายละเอียดเพิ่มเติม Drone" in msg:
            # Display quick reply options for drones
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label=name, text=f"รายละเอียดเพิ่มเติม {name}")) for name in ["Betafpv", "GEPRC", "FLYFISH RC", "Happymodel", "iFlight", "Team Black Sheep", "ToolkitRC"]
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="กรุณาเลือกเพิ่มเติม:", quick_reply=quick_reply))

        elif "รายละเอียดเพิ่มเติม Accessories" in msg:
            # Display quick reply options for accessories
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label=name, text=f"รายละเอียดเพิ่มเติม {name}")) for name in ["6s-batt", "กล้อง", "กระดองครอบ", "เฟรม", "บอร์ด", "ตัวรับสัญญาณ", "ตัวส่งสัญญาณภาพ", "มอเตอร์", "ใบพัด", "อุปกรณ์ชาร์จ", "เครื่องมือจำเป็น"]
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="กรุณาเลือกเพิ่มเติม:", quick_reply=quick_reply))


        if "ค้นหา" in msg:
            search_term = msg.replace("ค้นหา", "").strip()
            reply_text = "ต้องการเรทราคาต่ำกว่าเท่าไหร่ครับ (ต่ำกว่า+จำนวนเงิน)\nหรือเลือกแสดงทั้งหมดครับ"
            
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label="แสดงทั้งหมด", text="แสดงทั้งหมด")),
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            
            line_bot_api.reply_message(tk, TextSendMessage(text=reply_text, quick_reply=quick_reply))
                        
        

        if "เรียงลำดับราคา" in msg:
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label="ถูกสุดไปแพงสุด", text="ถูกสุดไปแพงสุด")),
                QuickReplyButton(action=MessageAction(label="แพงสุดไปถูกสุด", text="แพงสุดไปถูกสุด")),
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="กรุณาเลือกเพิ่มเติม : ", quick_reply=quick_reply))

        elif "ถูกสุดไปแพงสุด" in msg or "แพงสุดไปถูกสุด" in msg:
            sort_order = "asc" if "ถูกสุดไปแพงสุด" in msg else "desc"
            product_info = fetch_product_info(search_term)

            if product_info is not None:
                def clean_price(price_str):
                    price_str = price_str.replace('฿', '').replace(',', '').strip()
                    return float(price_str)

                # ตรวจสอบว่าผู้ใช้เลือก "ต่ำกว่า" หรือไม่
                if is_lower_selected:
                    # ใช้ price_min ที่เก็บค่าจากข้อความ "ต่ำกว่า"
                    filtered_products = [
                        item for item in product_info
                        if item['price'] != "Price not available" and clean_price(item['price']) < int(price_min)
                    ]
                else:
                    filtered_products = [
                        item for item in product_info
                        if item['price'] != "Price not available"
                    ]

                # การเรียงลำดับ
                if sort_order == "asc":
                    sorted_products = sorted(filtered_products, key=lambda x: clean_price(x['price']))
                else:
                    sorted_products = sorted(filtered_products, key=lambda x: clean_price(x['price']), reverse=True)

                response_msg = (
                    "\n\n".join(
                        [f"• ชื่อสินค้า: {item['title']}\n  ราคา: {item['price']}\n  ลิงค์: {item['link']}\n"
                            for item in sorted_products]
                    ) if sorted_products else "ไม่พบข้อมูลสินค้า"
                )

                line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณครับ"))

        if "ต่ำกว่า" in msg:
            msg = msg.replace("ต่ำกว่า", "").replace("ประมาณ", "").strip()
            price_min = re.findall(r'\d+', msg)
            price_min = ''.join(price_min)  # รวมตัวเลขที่เจอให้เป็นสตริงเดียวกัน
            is_lower_selected = True  # อัพเดตสถานะว่า "ต่ำกว่า" ถูกเลือก

            product_info = fetch_product_info(search_term)
            if product_info is not None:
                def clean_price(price_str):
                    price_str = price_str.replace('฿', '').replace(',', '').strip()
                    return float(price_str)

                response_msg = (
                    "\n\n".join(
                        [
                            f"• ชื่อสินค้า: {item['title']}\n  ราคา: {item['price']}\n  ลิงค์: {item['link']}\n"
                            for item in product_info 
                            if item['price'] != "Price not available" and clean_price(item['price']) < int(price_min)
                        ]
                    ) if product_info else "ไม่พบข้อมูลสินค้า"
                )

                if response_msg:
                    quick_reply_options = [
                        QuickReplyButton(action=MessageAction(label="แสดงทั้งหมด", text="แสดงทั้งหมด")),
                        QuickReplyButton(action=MessageAction(label="เรียงลำดับราคา", text="เรียงลำดับราคา")),
                    ]
                    quick_reply = QuickReply(items=quick_reply_options)

                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
                else:
                    line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณครับ"))

            elif product_info == None:
                line_bot_api.reply_message(tk , TextSendMessage(text='ขออภัยไม่มีสินค้า ที่ตรงกับความต้องการของคุณ'))
                
        if "แสดงทั้งหมด" in msg:
            product_info = fetch_product_info(search_term)
            is_lower_selected = False  # อัพเดตสถานะว่า "ต่ำกว่า" ถูกเลือก

            if product_info is not None:
                response_msg = (
                    "\n\n".join(
                        [
                            f"• ชื่อสินค้า: {item['title']}\n  ราคา: {item['price']}\n  ลิงค์: {item['link']}\n"
                            for item in product_info 
                            if item['price'] != "Price not available"
                        ]
                    ) if product_info else "ไม่พบข้อมูลสินค้า"
                )

                if response_msg:
                    quick_reply_options = [
                        QuickReplyButton(action=MessageAction(label="เรียงลำดับราคา", text="เรียงลำดับราคา")),
                    ]
                    quick_reply = QuickReply(items=quick_reply_options)
                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
                else:
                    line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณครับ"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณครับ"))

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
                print(response_msg)
                payload = {
                    "model": "supachai/llama-3-typhoon-v1.5",
                    "prompt": f"ผู้ตอบเป็นผู้เชี่ยวชาญเรื่องโดรน ผู้ถามชื่อ คุณ{user_name} ตอบสั้นๆไม่เกิน 20 คำ เกี่ยวกับ '{msg}'",
                    "stream": False
                }
                response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    decoded_text = response.json().get("response", "")
                    line_bot_api.reply_message(tk, TextSendMessage(text=decoded_text + 'ครับ'))
                    save_response(uid, msg, decoded_text)  # Save the answer and response
                else:
                    print(f"Failed to get a response from Ollama: {response.status_code}, {response.text}")
                    line_bot_api.reply_message(tk, TextSendMessage(text="เกิดข้อผิดพลาดในการติดต่อ LLaMA"))

    except InvalidSignatureError:
        return jsonify({'message': 'Invalid signature!'}), 400

    return jsonify({'status': 'OK'}), 200

if __name__ == "__main__":
    app.run(port=5000)
