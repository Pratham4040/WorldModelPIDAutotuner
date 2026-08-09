import requests
import time

class ESP32Client:
    """
    HTTP client to interface with the ESP32 microcontroller thermal chamber.
    Uses persistent HTTP connection pooling (requests.Session) to eliminate 
    connection timeouts over Wi-Fi.
    Endpoints:
      - GET /temp -> returns temperature (float) or error status
      - POST /pwm -> posts raw PWM value (0..255) in plain text body
      - GET /status -> returns JSON status
    """
    def __init__(self, ip_address, timeout=2.5):
        self.ip_address = ip_address
        self.base_url = f"http://{ip_address}"
        self.timeout = timeout
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=5, max_retries=1)
        self.session.mount("http://", adapter)

    def read_temp(self):
        """
        Queries GET /temp.
        Returns:
            (temp_float, safety_active_bool)
        """
        url = f"{self.base_url}/temp"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                print(f"[Warning] HTTP Error {response.status_code} on read_temp")
                return None, True
            
            text = response.text.strip()
            
            # Handle special return strings:
            # e.g., "nan,SENSOR_ERROR" or "39.54,SAFETY" or plain "23.4567"
            if "," in text:
                parts = text.split(",")
                temp_val = parts[0]
                status = parts[1]
                
                if status == "SENSOR_ERROR" or temp_val.lower() == "nan":
                    return None, True
                
                try:
                    return float(temp_val), (status == "SAFETY")
                except ValueError:
                    return None, True
            else:
                try:
                    return float(text), False
                except ValueError:
                    return None, True
                    
        except requests.exceptions.RequestException as e:
            print(f"[Error] Connection failed in read_temp: {e}")
            return None, True

    def set_pwm(self, pwm):
        """
        Sends POST /pwm with the raw integer body.
        Returns:
            (success_bool, returned_pwm_int)
        """
        url = f"{self.base_url}/pwm"
        pwm = max(0, min(255, int(pwm)))
        try:
            # Send raw text body
            response = self.session.post(url, data=str(pwm), headers={"Content-Type": "text/plain"}, timeout=self.timeout)
            if response.status_code != 200:
                print(f"[Warning] HTTP Error {response.status_code} on set_pwm")
                return False, 0
            
            text = response.text.strip()
            # Expect "OK:<pwm>" or similar
            if text.startswith("OK:"):
                try:
                    ret_pwm = int(text.split(":")[1])
                    return True, ret_pwm
                except (ValueError, IndexError):
                    return True, pwm
            return True, pwm
            
        except requests.exceptions.RequestException as e:
            print(f"[Error] Connection failed in set_pwm: {e}")
            return False, 0

    def get_status(self):
        """
        Queries GET /status.
        Returns:
            dict with 'temp', 'pwm', and 'safe', or None if failed.
        """
        url = f"{self.base_url}/status"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            print(f"[Error] Connection failed in get_status: {e}")
            return None
