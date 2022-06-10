import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'                                        #path for folder where images are stored
images = []                                                      #creating empty list of images
personNames = []                                                 #storing names of people after omitting jpg
myList=os.listdir(path)                                          #get list of all files and directories


# importing images one by one in the above lists
for cu_img in myList :
    current_Img=cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])


print('images imported')


#finding encodings of the images in the dataset
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#storing attendance in csv file
def attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime("%H:%M:%S")
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


#calling function findEncoding
encodeListKnown = findEncodings(images)
print('All Encodings Completed')

#initialising webcam
cap=cv2.VideoCapture(0)

#loop for capturing images one by one
while True:
    ret ,frame = cap.read()
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)                            #reducing the size of the image
    faces= cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)                           #changing color from BGR to RGB

    #detecting the faces location from frame(webcam) and encoding it
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame)

 #comparing faces taken from the webcam with the faces in the data set
    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)

        #if match found setting the rectangle frame
        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),1)
            attendance(name)                           #recording the attendance in csv file

        cv2.imshow('Webcam',frame)
        cv2.waitKey(1)
