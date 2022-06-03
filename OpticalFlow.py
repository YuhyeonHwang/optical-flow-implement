import cv2
import numpy as np
import copy
import time

epsilon = 0.1

def imgResize(img,rate=0.5):
    return cv2.resize(img, dsize=(0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)

def getDiff(img):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width = img.shape
    dy = np.zeros(shape=img.shape)
    dx = np.zeros(shape=img.shape)
    for y in range(height-1):
        for x in range(width-1):
            dy[y,x] = np.int16(img[y+1,x]) - np.int16(img[y,x])
    for y in range(height-1):
        for x in range(width-1):
            dx[y,x] = np.int16(img[y,x+1]) - np.int16(img[y,x])
    return dy, dx

def getDiff_t(img0,img1):
    if len(img0.shape)==3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    height,width = img0.shape
    dt = np.zeros(shape=img0.shape)
    for y in range(height-1):
        for x in range(width-1):
            dt[y,x] = np.int16(img1[y,x]) - np.int16(img0[y,x])
    return dt

def lstSqs(A,b):
    return np.dot(np.linalg.pinv(np.dot(A.T,A)),np.dot(A.T,b))

def optFlow(img0,img1,rSize=3):
    '''
    received t0 image, t1 image, region size : (rSize*2+1)**2
    return motion vector based on Lucas-Kanade algorithm
    '''
    if len(img0.shape)==3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    height,width = img0.shape
    dy,dx = getDiff(img0)
    dt = getDiff_t(img0,img1)
    v = np.zeros(shape=img0.shape,dtype=np.float32)
    u = np.zeros(shape=img0.shape,dtype=np.float32)
    w = rSize//2
    for y in range(w,height-w):
        for x in range(w,width-w):
            dyValue = dy[y-w:y+w+1,x-w:x+w+1]
            dxValue = dx[y-w:y+w+1,x-w:x+w+1]
            b = -dt[y-w:y+w+1,x-w:x+w+1].reshape(-1)
            A = np.vstack([dyValue.reshape(-1),dxValue.reshape(-1)]).T
            V = lstSqs(A,b)
            v[y,x],u[y,x] = V
    motionVec = np.array([v,u])/255
    return motionVec

def optFlowVis(img,motionVec,arrowedSkip=2,threshold=0.02,mulScale=1):
    '''
    received image, motion vetor, arrowedSkip, threshold, mulScale
    arrowedSkip means step size for print arrowedLine
    threshold arg is to suppress small values of motion vector
    mulScale arg is up size motion vector
    '''
    copyImg = img.copy()
    if len(img.shape)==3:
        imgShape = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape)==2:
        imgShape = img
    height,width = imgShape.shape
    diagonal = np.sqrt(np.square(height)+np.square(width))
    for y in range(0,height,arrowedSkip):
        for x in range(0,width,arrowedSkip):
            ### arrowedLine received args point[x,y]
            if np.sqrt(np.square(motionVec[0][y,x])+np.square(motionVec[1][y,x]))>threshold:
                img = cv2.arrowedLine(copyImg,[x,y],
                                      [np.int16(x+motionVec[0][y,x]*diagonal*mulScale),
                                       np.int16(y+motionVec[1][y,x]*diagonal*mulScale)],
                                        color=(0,0,255),thickness=1,line_type=4,shift=0,tipLength=0.2)
    return img

def main():
    img0 = cv2.imread('/home/hyh/robot_vision/cap_img/t_0.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('/home/hyh/robot_vision/cap_img/t_1.jpg', cv2.IMREAD_COLOR)
    # img1 = img0

    # img0 = imgResize(img0,rate=0.05)
    # img1 = imgResize(img1,rate=0.05)

    motionVec = optFlow(img0,img1,rSize=10)
    optFlowImg = optFlowVis(img0,motionVec,arrowedSkip=6,threshold=0.0005)

    cv2.imshow('original image',img0)
    cv2.imshow('optical flow image',optFlowImg)
    cv2.waitKey(0)

if __name__=='__main__':
    main()
