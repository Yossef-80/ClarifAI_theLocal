


def send_engagements_to_backend(self,engagements):
    import json
    import requests

    url = "http://your-backend-url/api/engagements"  # Replace with your actual URL
    headers = {'Content-Type': 'application/json'}
    payload = {"engagements": self.engagements}
    json_data = json.dumps(payload)

    try:
        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            print("Data sent successfully!")
        else:
            print("Failed to send data:", response.status_code, response.text)
    except Exception as e:
        print("Error sending data:", e)