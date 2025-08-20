from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from config import LINE_CONFIG

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CONFIG['channel_access_token'])
handler = WebhookHandler(LINE_CONFIG['channel_secret'])

@app.route("/callback", methods=['POST'])
def callback():
    # รับ signature จาก LINE
    signature = request.headers['X-Line-Signature']
    
    # รับข้อความ
    body = request.get_data(as_text=True)
    print("Request body: " + body)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature")
        abort(400)
        
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # เมื่อมีข้อความเข้ามา ตอบกลับพร้อม user ID
    user_id = event.source.user_id
    reply_text = f"Your LINE User ID is: {user_id}\nPlease add this ID to config.py"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

if __name__ == "__main__":
    # รัน webhook server
    app.run(debug=True, port=5000)
