import numpy as np
import cv2
import subprocess
import time
from face_detector2 import getFaces
from chainer import serializers
import subprocess

#from clasify2 import test
#from clasify2 import VGGNet
#model=VGGNet()
#serializers.load_hdf5("./face_recognition/VGG11_00226621233099.model",model)##input model path
#mean=np.load("./face_recognition/mean.npy")##input mean path

from clasify import test
from clasify import VGGNet
model=VGGNet()
serializers.load_hdf5("./face_recognition/VGG11_0223096959822.model",model)##input model path
mean=np.load("./face_recognition/mean.npy")##input mean path
nantoka = 'nigehaji'
input_video = './'+nantoka+'.mp4'
skip=40#if you want otoarimovie, fps / skip must be integer
testnum = 0
otoarimovie = False
output1 = "./pvbws"+str(skip)+nantoka+str(otoarimovie)+str(testnum)+".avi"
output2 = "./pvbws"+str(skip)+nantoka+str(otoarimovie)+str(testnum)+".mp4"
colect_face=True

def detect_face(image):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print "Found {0} faces!".format(len(faces))
    results=[]
    for (x, y, w, h) in faces:
        results.append(cv2.resize(image[y : y+h, x : x+w],(96,96)))
#    # Draw a rectangle around the faces
#    for (x, y, w, h) in faces:
#        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return results
def export_movie():
    global skip
    #target movie
    cap = cv2.VideoCapture(input_video)
    frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#	frame_number = 300
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    size = ((int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fps = (int)(cap.get(cv2.cv.CV_CAP_PROP_FPS)+0.5)
    print "fps={0}".format(fps)
    if fps % skip != 0 and otoarimovie == True:
        print "fps % skip != 0,error"
        cap.release()
        return False
    # open output
    out = cv2.VideoWriter(output1, fourcc, fps/skip, size)
    for i in xrange(frame_number):
        print "{0} / {1}".format(i,frame_number)
        ret1, frame = cap.read()
        if ret1==True and i % skip == 0:
            #results = detect_face(frame)
            results, leftbottoms = getFaces(frame)
            if len(results) > 0:
                if collect_face ==False:
                    ret2, strings = test(results, model, mean)
                        for j in range(len(results)):
                            cv2.putText(frame, text=strings[j], org=leftbottoms[j], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=1) 
                else:
                    for j in range(len(results)):
                        cv2.imwrite("./testdayozenninnsyuugou/frame"+str(i)+"kao"+str(j)+".jpg", results[j])
            else:
                cv2.putText(frame, text="There Is None", org=(size[0]/3,size[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,0,255), thickness=1) 
            # write the flipped frame
            if colect_face == False:
                out.write(frame)
            cv2.imshow('frame1',frame)
            k = cv2.waitKey(1)
            if k==27:
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True
def add_audio():
    cmd='ffmpeg -i '+output1+' -i '+input_video+' -vcodec copy -acodec copy '+output2
    subprocess.call(cmd, shell=True)
if __name__ == '__main__':
    start = time.time()
    ret=export_movie()
    if otoarimovie==True and ret==True:
        add_audio()
    print time.time() - start


