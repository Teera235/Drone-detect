import os
import requests
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage
from linebot.exceptions import LineBotApiError
import logging

class LineNotifier:
    def __init__(self, channel_access_token):
        """
        Initialize Line notifier
        channel_access_token: Line Messaging API channel access token
        """
        self.line_bot_api = LineBotApi(channel_access_token)
        self.logger = logging.getLogger("ORDNANCE_DRONE_DETECTION")
        
    def send_message(self, user_id: str, message: str):
        """ส่งข้อความ"""
        try:
            self.line_bot_api.push_message(
                user_id,
                TextSendMessage(text=message)
            )
            self.logger.info(f"[LINE] Sent message: {message}")
        except LineBotApiError as e:
            self.logger.error(f"[LINE] Failed to send message: {e}")
            
    def send_image(self, user_id: str, image_path: str, message: str = None):
        """ส่งรูปภาพพร้อมข้อความ (ถ้ามี)"""
        try:
            # ส่งรูปภาพ
            image_url = self._upload_image(image_path)
            if image_url:
                self.line_bot_api.push_message(
                    user_id,
                    ImageSendMessage(
                        original_content_url=image_url,
                        preview_image_url=image_url
                    )
                )
                self.logger.info(f"[LINE] Sent image: {image_path}")
                
                # ส่งข้อความถ้ามี
                if message:
                    self.send_message(user_id, message)
                    
        except LineBotApiError as e:
            self.logger.error(f"[LINE] Failed to send image: {e}")
            
    def _upload_image(self, image_path: str) -> str:
        """
        อัพโหลดรูปไปที่ imgbb.com
        """
        try:
            with open(image_path, "rb") as file:
                files = {"image": file}
                params = {"key": "0f9b9f150750305cde2fdf25dc933fbb"}  # imgbb.com API key
                
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    params=params,
                    files=files
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["data"]["url"]
                else:
                    self.logger.error(f"[LINE] Image upload failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            self.logger.error(f"[LINE] Failed to upload image: {e}")
            
        return None
