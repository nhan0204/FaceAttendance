import pandas as pd
import cv2
import urllib.request 
import numpy as np
import os
from datetime import datetime
import face_recognition

path = r'C:/Users/phamt/OneDrive/Documents/Code/PlatformIO/Projects/FaceAttendance/attendance/image_folder'
url='http://192.168.3.7:81/800x600.jpg'
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''

if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd(),'attendance')):
    print("there iss..")
    os.remove("Attendance.csv")
else:
    df=pd.DataFrame(list())
    df.to_csv("Attendance.csv")
    
 
images = []
classNames = []
imgList = os.listdir(path)
print(imgList)
for classMate in imgList:
    currentImg = cv2.imread(f'{path}/{classMate}')
    images.append(currentImg)
    classNames.append(os.path.splitext(classMate)[0])
print(classNames)
 
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
 
def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()      
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
 
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
#cap = cv2.VideoCapture(0)
 
while True:
    #success, img = cap.read()
    img_respone = urllib.request.urlopen(url)
    img_np = np.array(bytearray(img_respone.read()), dtype = np.uint8)
    img = cv2.imdecode(img_np, -1)
# img = captureScreen()
    imgSource = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSource = cv2.cvtColor(imgSource, cv2.COLOR_BGR2RGB)
 
    facesCurrentFrame = face_recognition.face_locations(imgSource)
    encodesCurrentFrame = face_recognition.face_encodings(imgSource, facesCurrentFrame)
 
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)
 
        if not(matches[matchIndex]):
            print('UNKNOWN')
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'UNKNOWN', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
            
 
    cv2.imshow('Webcam', img)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
cv2.destroyAllWindows()
cv2.imread