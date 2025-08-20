import socket
import json
import logging
from typing import Optional, Dict, Any

class WiFiController:
    """Controller for sending control commands over WiFi"""
    
    def __init__(self, host: str = "192.168.4.1", port: int = 80):
        """Initialize WiFi controller
        
        Args:
            host: IP address of ESP32 controller (default: 192.168.4.1) 
            port: Port number (default: 80)
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.logger = logging.getLogger("ORDNANCE_DRONE_DETECTION")
        self._connect()
        
    def _connect(self) -> None:
        """Establish socket connection to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1.0)  # 1 second timeout
            self.socket.connect((self.host, self.port))
            self.logger.info(f"[WIFI] Connected to {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"[WIFI] Connection failed: {e}")
            self.socket = None
            
    def send_data(self, data: Dict[str, Any]) -> bool:
        """Send control data over WiFi
        
        Args:
            data: Dictionary of control data to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.socket:
            try:
                self._connect()
            except:
                return False
                
        try:
            msg = json.dumps(data).encode() + b"\n"
            self.socket.send(msg)
            return True
        except Exception as e:
            self.logger.error(f"[WIFI] Send failed: {e}")
            self.socket = None
            return False
            
    def close(self) -> None:
        """Close the socket connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
