import requests

try:
    # First get token using the admin account (kaushal@gmail.com)
    resp = requests.post("https://quant-backend-w1lf.onrender.com/auth/login", json={
        "email": "kaushal@gmail.com",
        "password": "Kaushal@2004"
    }, timeout=20)
    token = resp.json().get("access_token")
    
    if token:
        # Now get the user ID for kaushalnandania086@gmail.com
        users_resp = requests.get("https://quant-backend-w1lf.onrender.com/admin/users", 
                                  headers={"Authorization": f"Bearer {token}"}, timeout=20)
        users = users_resp.json().get("users", [])
        
        target_id = None
        for u in users:
            if u["email"] == "kaushalnandania086@gmail.com":
                target_id = u["id"]
                break
                
        if target_id:
            print(f"Found target user ID: {target_id}")
            
            # Update the role
            update_resp = requests.put(f"https://quant-backend-w1lf.onrender.com/admin/users/{target_id}/role",
                                       headers={"Authorization": f"Bearer {token}"},
                                       json={"role": "admin"}, timeout=20)
            print("Update response:", update_resp.status_code, update_resp.text)
        else:
            print("Target user not found")
    else:
        print("Failed to get admin token")
        
except Exception as e:
    print("Error:", e)

