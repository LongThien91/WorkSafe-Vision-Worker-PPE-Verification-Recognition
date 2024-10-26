from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import face_recognition
from numpy.ma.core import ceil

modelSegment = YOLO("SegmentFinalMLast.pt") 
modelDetect = YOLO("DetectFinalMLast.pt")

mode=0
face_on=0
while True:
    try:
        mode = int(input("Choose detection mode: \n0 - Use camera\n1 - Use video path\n"))
        if mode==0 or mode==1:
            break
    except:
        continue

while True:
    try:
        face_on = int(input("Turn on Face Recognition Mode?: \n0 - No\n1 - Yes\n"))
        if face_on==0 or face_on==1:
            break
    except:
        continue
if(face_on==1):
    print("start encode face ...")
    encodeID=[]
    listIDImagesPath=[f for f in os.listdir('D:\Project\WorkSafe-Vision-Worker-PPE-Verification-Recognition\WorkerFaceImage') if f.endswith(('.jpg', '.jpeg', '.png'))]
    IDs=[]
    for path in listIDImagesPath:
        imgPath='D:\Project\WorkSafe-Vision-Worker-PPE-Verification-Recognition\WorkerFaceImage/'+path
        imageID=cv2.imread(imgPath)
        imageID=cv2.cvtColor(imageID,cv2.COLOR_BGR2RGB)
        #imageID=cv2.resize(imageID,(512,512))
        facelocate=face_recognition.face_locations(imageID)
        if len(facelocate)>0:
            encode=face_recognition.face_encodings(imageID,facelocate)[0]
            IDs.append( path[:-4])
            encodeID.append(encode)
            print(path)
    print("Finish")
cap=0
if mode==1:
    videoPath=input('Enter your video path here:')
    cap=cv2.VideoCapture(videoPath)
else:
    cap=cv2.VideoCapture(0)
print('press q to quit')
while True:
    ret, frame= cap.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(512,512))
    results=modelSegment.predict(source=frame,conf=0.6, save=False,verbose=False)
    for i in range(0,len(results[0])):
        #cv2_imshow((results[0].masks.data[i].numpy()*255).astype("uint8"))
        mask=results[0].masks.cpu().data[i].numpy()
        mask_image = frame.copy()
        mask_image[mask == 0] = [255, 255, 255]
        color_img=frame.copy()
        color_img[mask == 0] = [255, 255, 255]
        color_img[mask == 1] = [0, 255, 0]
        frame[mask == 1]=cv2.addWeighted(frame[mask == 1],0.5,color_img[mask == 1],0.5,0)
        if face_on==1:
            facelocate=face_recognition.face_locations(mask_image)
            if len(facelocate)>0 :
                encode=face_recognition.face_encodings(mask_image,facelocate)
                for encodeFace, faceLoc in zip(encode,facelocate):
                    matches = face_recognition.compare_faces(encodeID,encodeFace)
                    faceDis = face_recognition.face_distance(encodeID,encodeFace)
                    matchIndex=np.argmin(faceDis)
                    print("matches",matchIndex)
                    print('dis',faceDis[matchIndex])

                    y1,x2,y2,x1 = facelocate[0]
                    frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                    cv2.putText(frame,f'{IDs[matchIndex]}',(x1,y1),cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,0,255),thickness=1)
        
        resDetect= modelDetect.predict(mask_image,conf=0.5,verbose=False)
        for r in resDetect:
            boxes=r.boxes
            for box in boxes:
              x1,y1,x2,y2= box.xyxy[0]
              x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
              w,h=x1-x2,y1-y2
              conf=ceil(float(box.conf[0])*100)/100
              labelBox=modelDetect.names.get(box.cls.item())
              if(labelBox=='not_helmet' or labelBox =='not_safetyVest'):
                frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(frame,f'{labelBox}  {conf}',(x1,y1),cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,0,255),thickness=1)
              else:
                frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame,f'{labelBox}  {conf}',(x1,y1),cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,255,0),thickness=1)
    cv2.imshow("Image",frame)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    