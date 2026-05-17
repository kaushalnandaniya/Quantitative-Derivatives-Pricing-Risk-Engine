import requests
import time

try:
    # First get token using the admin account (kaushal@gmail.com)
    resp = requests.post("https://quant-backend-w1lf.onrender.com/auth/login", json={
        "email": "kaushal@gmail.com",
        "password": "Kaushal@2004"
    }, timeout=20)
    token = resp.json().get("access_token")
    
    if token:
        target_id = "d86c63d1-d3d1-4102-aa5d-1f35eaaefc25" # Hardcoded from previous run
        print(f"Target user ID: {target_id}")
        
        # Retry logic for the update since Render can drop connections during cold starts
        for attempt in range(3):
            try:
                print(f"Update attempt {attempt + 1}...")
                update_resp = requests.put(f"https://quant-backend-w1lf.onrender.com/admin/users/{target_id}/role",
                                           headers={"Authorization": f"Bearer {token}"},
                                           json={"role": "admin"}, timeout=30)
                print("Update response:", update_resp.status_code, update_resp.text)
                if update_resp.status_code == 200:
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed:", e)
                time.sleep(2)
    else:
        print("Failed to get admin token")
        
except Exception as e:
    print("Error:", e)

