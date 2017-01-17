import numpy as np
import cv2
import subprocess
import time
from face_detector2 import getFaces
from chainer import serializers

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

input_video = './nigehajikai.mp4'
testesnum = "kai"
output = "./pvbws_"+testesnum+".avi" 

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
    #target movie
    cap = cv2.VideoCapture(input_video)
    # form of output 
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = ((int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    # open output
    skip = 10
    out = cv2.VideoWriter(output, fourcc, fps/skip, size)
    i=0
    while(cap.isOpened()):
        print i
        i+=1
        ret1, frame = cap.read()
        if not ret1:
            print ret1
        if ret1==True:
            if i % skip == 0:
                #results = detect_face(frame)
                results, leftbottoms = getFaces(frame)
                if len(results) > 0:
                    ret2, strings = test(results, model, mean)
                    if True:#ret2:
                        for j in range(len(results)):
                            cv2.putText(frame, text=strings[j], org=leftbottoms[j], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=1) 
    #                    for j in xrange(len(results)):
    #                        cv2.imwrite("./kaokai/frame"+str(i)+"kao"+str(j)+".jpg", results[j])
                        print "write"
                else:
                    cv2.putText(frame, text="There Are None", org=(size[0]/3,size[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,0,255), thickness=1) 
                # write the flipped frame
                out.write(frame)
                cv2.imshow('frame1',frame)
                k = cv2.waitKey(1)
                if k==27:
                    break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start = time.time()
    export_movie()
    print time.time() - start


