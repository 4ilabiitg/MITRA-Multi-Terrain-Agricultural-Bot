from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import os
from ultralytics import YOLO
import math
import requests
import json
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

app = Flask(__name__)
# Load the YOLO model
model = YOLO("mitra.pt")
# Object classes
classNames = ['Crop', 'Weed']

# Setup Ollama with gemma model
ollama_llm = Ollama(model="gemma:2b")  # You can also use "gemma:7b" for better results

# Setup web search tool
search_tool = DuckDuckGoSearchRun()

def generate_frames():
    # Replace 'path/to/your/video.mp4' with the actual path to your video file
    cap = cv2.VideoCapture('mitra.mp4')
    
    while True:
        success, img = cap.read()  # Read a frame from the video

        if not success:  # If there are no more frames, restart the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Perform detection
        results = model(img, stream=True)  # Run YOLO detection

        # Process the detection results and draw bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                if cls == 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                org = (x1, y1 + 60)

                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'{classNames[cls]} {confidence:.2f}'
                font_scale = 2  
                thickness = 3
                text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
                cv2.putText(img, text, org, font, font_scale, (210, 16, 16), thickness, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

def get_answer_with_search(query):
    """Generate answer with web search capability"""
    # First, try to find relevant information online
    try:
        search_results = search_tool.run(f"agriculture farming {query}")
    except Exception as e:
        search_results = f"Search error: {str(e)}"
    
    # Craft a prompt that includes search results and proper formatting
    farmer_prompt = f"""You are an agricultural expert AI assistant named Mitra AI. A farmer has asked: "{query}"

Search results found:
{search_results}

Please provide helpful information to the farmer based on the search results and your knowledge.
Format your response in this structured way:

1. Direct Answer: [Brief 1-2 sentence direct answer]
2. Explanation: [More detailed explanation of the solution]
3. Implementation: [Step by step practical advice]
4. Additional Tips: [1-3 bullet points with extra tips]

Only answer agricultural queries. If the question is not related to agriculture, reply that you can only assist with farming topics.
Don't use markdown formatting or special characters - just plain text with numbered sections and clean formatting.
"""
    
    # Generate the response using Ollama
    response = ollama_llm.invoke(farmer_prompt)
    return response

@app.route('/query', methods=['POST'])
def handle_query():
    query = request.form['query']
    
    # Get answer with web search capability
    ans = get_answer_with_search(query)
    
    # Clean any potential formatting issues
    ans = ans.replace('**', '')
    ans = ans.replace('*', '')
    
    print("Response:", ans)
    return render_template('index.html', query=query, response=ans)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)