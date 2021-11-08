import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


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


def roberts(img):
    img=np.array(img)
    out=img.copy()
    for i in range(out.shape[0]-1):
        for j in range(out.shape[1]-1):
            template1=np.array([[-1,0],[0,1]])
            template2=np.array([[0,-1],[1,0]])
            out[i,j]=math.sqrt(abs(np.sum(img[i:i+2,j:j+2]*template1))**2+abs(np.sum(img[i:i+2,j:j+2]*template2))**2)
    return cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)[1]


def prewitt(img):
    img=np.array(img)
    out=img.copy()
    for i in range(out.shape[0]-2):
        for j in range(out.shape[1]-2):
            template1=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            template2=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            out[i,j]=math.sqrt(abs(np.sum(img[i:i+3,j:j+3]*template1))**2+ abs(np.sum(img[i:i+3,j:j+3]*template2))**2)
    return cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)[1]


def sobel(img):
    img=np.array(img)
    out=img.copy()
    for i in range(out.shape[0]-2):
        for j in range(out.shape[1]-2):
            template1=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
            template2=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
            out[i,j]=math.sqrt(abs(np.sum(img[i:i+3,j:j+3]*template1))**2+abs(np.sum(img[i:i+3,j:j+3]*template2)))
    return cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)[1]

def cv_laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)  # 算子的大小，必须为1、3、5、7
    laplacian = cv2.convertScaleAbs(lap)  # 转回uint8
    return laplacian

def process(img):
    rob=roberts(img)
    # aa,contours, hierarchy=cv2.findContours(rob,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("img", aa)

    prew=prewitt(img)
    sob = sobel(img)
    #
    lap_img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    lap_img = cv2.convertScaleAbs(lap_img)
    Log = cv2.Laplacian(cv2.GaussianBlur(img, (5, 5), 1), cv2.CV_16S, ksize=3)
    Log = cv2.convertScaleAbs(Log)
    # 瓶盖Canny（ 50,80 ）   硬币Canny（ 270, 350 ）
    can = cv2.Canny(img, 270, 350 )

    plt.figure()
    ax=plt.subplot(231)
    plt.imshow(rob, cmap='gray')
    plt.axis('off')
    ax.set_title("roberts",size=25)


    ax=plt.subplot(232)
    plt.imshow(prew, cmap='gray')
    plt.axis('off')
    ax.set_title("prewitt",size=25)

    ax=plt.subplot(233)
    plt.imshow(sob, cmap='gray')
    plt.axis('off')
    ax.set_title("sobel",size=25)

    ax=plt.subplot(234)
    plt.imshow(lap_img, cmap='gray')
    plt.axis('off')
    ax.set_title("Laplacian",size=25)

    ax=plt.subplot(235)
    plt.imshow(Log, cmap='gray')
    plt.axis('off')
    ax.set_title("LoG",size=25)

    ax=plt.subplot(236)
    plt.imshow(can, cmap='gray')
    plt.axis('off')
    ax.set_title("Canny",size=25)

    # plt.show()
    plt.savefig("./result/1.png")


a = cv2.cvtColor(cv2.imread("./2.png"), cv2.COLOR_RGB2GRAY)
# cv2.imshow("img", a)
# a = cv2.cvtColor(cv2.imread("./Xray/capacitor.jpg") , cv2.COLOR_RGB2GRAY)
process(a)