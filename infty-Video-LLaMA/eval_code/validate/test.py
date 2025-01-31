import json

# Input data (assuming this is the loaded JSON data)
data = [
    {"video_id": "001", "duration": "short", "domain": "Knowledge", "sub_category": "Humanity & History", 
     "url": "https://www.youtube.com/watch?v=fFjv93ACGo8", "videoID": "fFjv93ACGo8", "question_id": "001-1", 
     "task_type": "Counting Problem", "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?", 
     "options": ["A. Apples.", "B. Candles.", "C. Berries.", "D. The three kinds are of the same number."], 
     "answer": "C"},
    {"video_id": "001", "duration": "short", "domain": "Knowledge", "sub_category": "Humanity & History", 
     "url": "https://www.youtube.com/watch?v=fFjv93ACGo8", "videoID": "fFjv93ACGo8", "question_id": "001-2", 
     "task_type": "Information Synopsis", "question": "What is the genre of this video?", 
     "options": ["A. It is a news report that introduces the history behind Christmas decorations.", 
                 "B. It is a documentary on the evolution of Christmas holiday recipes.", 
                 "C. It is a travel vlog exploring Christmas markets around the world.", 
                 "D. It is a tutorial on DIY Christmas ornament crafting."], 
     "answer": "A"}
]

# Transforming the data
result = {}
for entry in data:
    video_id = entry["video_id"]
    
    # Initialize the video entry in the result dictionary if not already
    if video_id not in result:
        result[video_id] = {
            "video_id": video_id,
            "duration": entry["duration"],
            "domain": entry["domain"],
            "sub_category": entry["sub_category"],
            "questions": []
        }
    
    # Append the question to the corresponding video entry
    question_data = {
        "question_id": entry["question_id"],
        "task_type": entry["task_type"],
        "question": entry["question"],
        "options": entry["options"],
        "answer": entry["answer"]
    }
    
    result[video_id]["questions"].append(question_data)

# Converting the result dictionary to the required list format
formatted_result = list(result.values())

# Output the formatted result
print(json.dumps(formatted_result, indent=4))