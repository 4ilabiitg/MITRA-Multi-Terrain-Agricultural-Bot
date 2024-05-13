from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import os
from ultralytics import YOLO
import math
import google.generativeai as genai 
from langchain_google_genai import ChatGoogleGenerativeAI
from  langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser



gemini_api_key = 'AIzaSyD4zSm-BKmKpKb5RabA-f8KqyvxPq3uWhA'
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.6, google_api_key=gemini_api_key)
os.environ['GOOGLE_API_KEY'] = gemini_api_key
genai.configure(api_key=gemini_api_key)
model1 = genai.GenerativeModel('gemini-pro')
chat = model1.start_chat(history=[])

app = Flask(__name__)
# Load the YOLO model
model = YOLO("mitra.pt")
# Object classes
classNames = ['Crop', 'Weed']

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

@app.route('/query', methods=['POST'])
def handle_query():
    query = request.form['query']
    # response = model1.generate_content(query)
    # ans = response.text
    # ans = ans.replace('**', ' ')
    # ans = ans.replace('*', '\n')
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template('You are an agricultual expert named Mitra Ai. your task is to give response to only agricultural queries not to any other and give the output such that it can be easily represented on a flask based website and dont do any formatting at all: {topic}')
    chain = prompt | llm | output_parser
    ans = chain.invoke({'topic':query})
    ans.replace('**', ' ')
    ans.replace('*', '')
    print(ans)
    return render_template('index.html', query=query, response=ans)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)