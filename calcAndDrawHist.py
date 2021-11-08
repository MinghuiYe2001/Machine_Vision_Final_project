import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out


def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)
    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg

def roberts(img):
    img=np.array(img)
    out=img.copy()
    for i in range(out.shape[0]-1):
        for j in range(out.shape[1]-1):
            template1=np.array([[-1,0],[0,1]])
            template2=np.array([[0,-1],[1,0]])
            out[i,j]=math.sqrt(abs(np.sum(img[i:i+2,j:j+2]*template1))**2+abs(np.sum(img[i:i+2,j:j+2]*template2))**2)

    hist = calcAndDrawHist(out, [255, 0, 0])
    return cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)[1],hist


def prewitt(img):
    img=np.array(img)
    out=img.copy()
    for i in range(out.shape[0]-2):
        for j in range(out.shape[1]-2):
            template1=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            template2=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            out[i,j]=math.sqrt(abs(np.sum(img[i:i+3,j:j+3]*template1))**2+ abs(np.sum(img[i:i+3,j:j+3]*template2))**2)

    hist = calcAndDrawHist(out, [255, 0, 0])
    return cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)[1],hist


def sobel(img):
    img=np.array(img)
    out=img.copy()
    for i in range(out.shape[0]-2):
        for j in range(out.shape[1]-2):
            template1=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
            template2=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
            out[i,j]=math.sqrt(abs(np.sum(img[i:i+3,j:j+3]*template1))**2+abs(np.sum(img[i:i+3,j:j+3]*template2)))

    hist = calcAndDrawHist(out, [255, 0, 0])
    return cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)[1],hist

def cv_laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)  # 算子的大小，必须为1、3、5、7
    laplacian = cv2.convertScaleAbs(lap)  # 转回uint8
    return laplacian

def process(img):
    _,rob=roberts(img)
    _, pre = prewitt(img)
    _, sol = sobel(img)


    plt.figure()
    ax=plt.subplot(131)
    plt.imshow(rob, cmap='gray')
    # plt.axis('off')
    ax.set_title("roberts",size=25)


    ax=plt.subplot(132)
    plt.imshow(pre, cmap='gray')
    # plt.axis('off')
    ax.set_title("prewitt",size=25)

    ax=plt.subplot(133)
    plt.imshow(sol, cmap='gray')
    # plt.axis('off')
    ax.set_title("soble",size=25)

    plt.show()
    plt.waitforbuttonpress()

    # plt.savefig("./result/1.png")



a = cv2.cvtColor(cv2.imread("./2.png"), cv2.COLOR_RGB2GRAY)
# cv2.imshow("img", a)
# a = cv2.cvtColor(cv2.imread("./Xray/capacitor.jpg") , cv2.COLOR_RGB2GRAY)
process(a)