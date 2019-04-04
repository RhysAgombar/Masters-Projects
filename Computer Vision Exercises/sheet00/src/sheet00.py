import cv2 as cv
import numpy as np
import random
import sys

if __name__ == '__main__':
    img_path = sys.argv[1]

    # 2a: read and display the image
    img = cv.imread(img_path)
    cv.imshow('2a',img)

    # 2b: display the intenstity image
    intensity = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:,:,2]
    cv.imshow('2b',intensity)

    # 2c: for loop to perform the operation
    img2 = np.copy(img)
    for j in range(0, img.shape[0]):
        for k in range(0, img.shape[1]):
            for i in range(0, img.shape[2]):
                img2[j,k,i] = (img2[j,k,i] - intensity[j,k] * 0.5).clip(min = 0)
    cv.imshow('2c',img2)

    # 2d: one-line statement to perfom the operation above
    img3 = np.copy(img)
    img3[:,:,:] =  (img3 - np.expand_dims(intensity,2) * 0.5).clip(min = 0)
    cv.imshow('2d',img3)

    # 2e: Extract a random patch
    img4 = np.copy(img)
    tl = (int(np.round(random.uniform(0,img4.shape[0] - 16))), int(np.round(random.uniform(0,img4.shape[1] - 16)))) #Generates a random point for the top left corner of the rectangle. Have to round and convert to integers to use this as an index.
    center = (int(np.round(img4.shape[0]/2)), int(np.round(img4.shape[1]/2)))

    img4[tl[0]:tl[0]+16, tl[1]:tl[1]+16, :] = img4[center[0]-8:center[0]+8, center[1]-8:center[1]+8, :]
    cv.imshow('2e',img4)

    # 2f: Draw random rectangles and ellipses
    img5 = np.copy(img)
    for i in range(0,10):
        sz = int(np.round(random.uniform(0,img4.shape[1]/5))) # random sizes for rectangle dimensions
        sz2 = int(np.round(random.uniform(0,img4.shape[0]/5)))
        tl = (int(np.round(random.uniform(0,img4.shape[1] - sz))), int(np.round(random.uniform(0,img4.shape[0] - sz2)))) # create top left point
        br = (tl[0] + sz2, tl[1] + sz) # Create bottom right point by adding sizes to top left point parameters
        img5 = cv.rectangle(img5,tl, br, (random.uniform(0, 255),random.uniform(0, 255),random.uniform(0, 255)), -1) # Draw rectangle using top left and bottom right points, and generate random colour

    # draw ellipses
    for i in range(0,10):
        sz = int(np.round(random.uniform(0,img4.shape[1]/5))) # Basically the same stuff as rectangles
        sz2 = int(np.round(random.uniform(0,img4.shape[0]/5)))
        center = (int(np.round(random.uniform(sz2,img4.shape[1] - sz2))), int(np.round(random.uniform(sz,img4.shape[0] - sz))))
        img5 = cv.ellipse(img5, center, (sz2, sz), 0, 0, 360, (random.uniform(0, 255),random.uniform(0, 255),random.uniform(0, 255)), -1)

    cv.imshow('2f',img5)

    # destroy all windows
    cv.waitKey(0)
    cv.destroyAllWindows()
