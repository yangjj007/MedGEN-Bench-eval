import json

def extract_json(select_response):
    select_response = select_response.replace('```json', '').replace('```', '')
    start_index = select_response.find('{')
    end_index = select_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = select_response[start_index:end_index + 1]
        return json.loads(json_str)
    else:
        return json.loads(select_response)