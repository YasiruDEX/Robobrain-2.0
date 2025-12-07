import urllib.request
import urllib.parse
import json
import time
import sys

BASE_URL = "https://h847mxpg-https://h847mxpg-5001.inc1.devtunnels.ms/.inc1.devtunnels.ms//api"

def test_health():
    print("Testing /health...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health") as response:
            data = json.loads(response.read().decode())
            print(f"Health check passed: {data}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_tasks():
    print("Testing /tasks...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/tasks") as response:
            data = json.loads(response.read().decode())
            print(f"Tasks check passed: {data}")
            return True
    except Exception as e:
        print(f"Tasks check failed: {e}")
        return False

def test_session():
    print("Testing /session...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/session", method="POST")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"Session created: {data}")
            return data.get("session_id")
    except Exception as e:
        print(f"Session creation failed: {e}")
        return None

def test_chat(session_id):
    print(f"Testing /chat for session {session_id}...")
    payload = {
        "session_id": session_id,
        "message": "Hello, are you there?",
        "task": "general",
        "enable_thinking": False
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/chat", 
        data=data, 
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"Chat response: {result}")
            return True
    except Exception as e:
        print(f"Chat failed: {e}")
        return False

if __name__ == "__main__":
    print("Waiting for backend to start...")
    time.sleep(5) # Give backend some time to start if run concurrently
    
    if not test_health():
        sys.exit(1)
        
    if not test_tasks():
        sys.exit(1)
        
    session_id = test_session()
    if session_id:
        test_chat(session_id)
    else:
        sys.exit(1)
