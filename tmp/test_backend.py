import requests
import json

BASE_URL = "http://127.0.0.1:5003"

def test_backend():
    print("--- 1. Login as Admin ---")
    login_data = {"username": "admin", "password": "Admin@2025"}
    try:
        r = requests.post(f"{BASE_URL}/api/login", json=login_data)
        r.raise_for_status()
        token = r.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Success: Admin logged in.")
    except Exception as e:
        print(f"Error: Admin login failed: {e}")
        return

    print("\n--- 2. Get Model Info ---")
    try:
        r = requests.get(f"{BASE_URL}/api/model/info", headers=headers)
        r.raise_for_status()
        print(f"Success: Model Info: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Error: Model Info failed: {e}")

    print("\n--- 3. Get Settings ---")
    try:
        r = requests.get(f"{BASE_URL}/api/settings", headers=headers)
        r.raise_for_status()
        print(f"Success: Settings: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Error: Get Settings failed: {e}")

    print("\n--- 4. Update Settings ---")
    try:
        update_data = {"refresh_rate": "5", "notifications": False}
        r = requests.post(f"{BASE_URL}/api/settings", json=update_data, headers=headers)
        r.raise_for_status()
        print(f"Success: Update Settings: {r.json()['message']}")
    except Exception as e:
        print(f"Error: Update Settings failed: {e}")

    print("\n--- 5. List Users (Admin Only) ---")
    try:
        r = requests.get(f"{BASE_URL}/api/users", headers=headers)
        r.raise_for_status()
        print(f"Success: Found {len(r.json())} users.")
    except Exception as e:
        print(f"Error: List Users failed: {e}")

    print("\n--- 6. Access Stats (Admin Only) ---")
    try:
        r = requests.get(f"{BASE_URL}/api/stats/access", headers=headers)
        r.raise_for_status()
        print(f"Success: Access Stats: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Error: Access Stats failed: {e}")

    print("\n--- 7. Add User (Admin Only) ---")
    try:
        new_user = {
            "username": "test_user_ai",
            "name": "Test User AI",
            "email": "test@asgard.ai",
            "password": "TestPassword@2025",
            "role": "it_staff",
            "department": "IT",
            "pages": ["dashboard", "profile", "it"]
        }
        r = requests.post(f"{BASE_URL}/api/users/add", json=new_user, headers=headers)
        r.raise_for_status()
        print(f"Success: Added User: {r.json()['message']}")
    except Exception as e:
        print(f"Error: Add User failed: {e}")

    print("\n--- 8. Verification Complete ---")

if __name__ == "__main__":
    test_backend()
