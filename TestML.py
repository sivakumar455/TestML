

import json,requests


url = "http://localhost:5000/api"

data = json.dumps({'sl': 1.2, 'sw': 1.2, 'pl': 1.3, 'pw': 1.2})
r = requests.post(url, data)
print ("Hello")
print (r.json())