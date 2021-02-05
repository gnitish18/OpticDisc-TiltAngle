import os
import cv2
import xlwt
import time
import math
import random
import argparse
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage import color
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import active_contour
from skimage.filters import meijering, sato, frangi, hessian
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split 

def nothing(x):
    pass

def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    img = img.astype(np.uint8)
    cv2.imwrite(path,img)
    # Convert the scale (values range) of the image
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    # Save file
    plt.savefig(path, bbox_inches='tight')#, img, format = 'png')

def load_images_from_folder(folder):
    images = []
    filenames = []
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,fname))
        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
            filenames.append(fname)
    return images, filenames

def linear_regression(X_R,X_G):
    nl = X_R.shape[0]
    nr = X_G.shape[0]
  
    # mean of x and y vector 
    m_X_R, m_y_R = np.mean(X_R[:,0]), np.mean(X_R[:,1]) 
    m_X_G, m_y_G = np.mean(X_G[:,0]), np.mean(X_G[:,1]) 
  
    # calculating cross-deviation and deviation about X_R and X_G
    SS_xy_l = np.sum(X_R[:,1]*X_R[:,0]) - nl*m_y_R*m_X_R 
    SS_xx_l = np.sum(X_R[:,0]*X_R[:,0]) - nl*m_X_R*m_X_R 
    
    SS_xy_r = np.sum(X_G[:,1]*X_G[:,0]) - nr*m_y_G*m_X_G 
    SS_xx_r = np.sum(X_G[:,0]*X_G[:,0]) - nr*m_X_G*m_X_G 

    # calculating regression coefficients 
    bl = np.zeros((2))
    br = np.zeros((2))
    bl[1] = SS_xy_l / SS_xx_l
    bl[0] = m_y_R - bl[1]*m_X_R
    
    br[1] = SS_xy_r / SS_xx_r
    br[0] = m_y_G - br[1]*m_X_G

    return bl, br

def threshold_tb(image, threshold = 60, th_type = 3):
    i = image.copy()
    R_low = 0
    R_high = 255
    G_low = 0
    G_high = 255
    B_low = 0
    B_high = 255
    s = 0
    
    """
    cv2.namedWindow('Colour')
    cv2.createTrackbar('R_low','Colour', R_low, 255, nothing)
    cv2.createTrackbar('R_high','Colour', R_high, 255, nothing)
    cv2.createTrackbar('G_low','Colour', G_low, 255, nothing)
    cv2.createTrackbar('G_high','Colour', G_low, 255, nothing)
    cv2.createTrackbar('B_low','Colour', B_low, 255, nothing)
    cv2.createTrackbar('B_high','Colour', B_high, 255, nothing)

    #cv2.createTrackbar('Threshold','Contrast', threshold, 255, nothing)
    #cv2.createTrackbar('Type','Contrast', th_type, 4, nothing)

    cv2.createTrackbar('Switch', 'Contrast', s, 1, nothing)

    while(1):
        R_low = cv2.getTrackbarPos('R_low', 'Colour')
        R_high = cv2.getTrackbarPos('R_high', 'Colour')
        G_low = cv2.getTrackbarPos('G_low', 'Colour')
        G_high = cv2.getTrackbarPos('G_high', 'Colour')
        B_low = cv2.getTrackbarPos('B_low', 'Colour')
        B_high = cv2.getTrackbarPos('B_high', 'Colour')        
        #threshold = cv2.getTrackbarPos('Threshold','Contrast')
        #th_type = cv2.getTrackbarPos('Type','Contrast')
        s = cv2.getTrackbarPos('Switch','Contrast')
        
        
        
        #_, dst = cv2.threshold(image, 70, 255, 3)
        #i = cv2.convertScaleAbs(dst, alpha=alpha/100, beta=-1*beta)
        cv2.imshow('Rm', R_mask)
        cv2.imshow('Gm', G_mask)
        cv2.imshow('R', R_res)
        cv2.imshow('G', G_res)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

        #k = cv2.waitKey(1) & 0xFF
        #if k == 27:
        #    break
    """

    R_lower = np.array([0, 0, 150])
    R_higher = np.array([50, 50, 255])
    R_mask = cv2.inRange(i, R_lower, R_higher)
    R_res = cv2.bitwise_and(i, i, mask= R_mask)

    G_lower = np.array([0, 150, 0])
    G_higher = np.array([50, 255, 150])
    G_mask = cv2.inRange(i, G_lower, G_higher)
    G_res = cv2.bitwise_and(i, i, mask= G_mask)
    
    #R = cv2.cvtColor(R_mask, cv2.COLOR_BGR2GRAY)
    #G = cv2.cvtColor(G_mask, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('Rm', R_mask)
    #cv2.imshow('Gm', G_mask)
    #cv2.imshow('R', R_res)
    #cv2.imshow('G', G_res)
    #cv2.waitKey(0)
    
    #i = cv2.convertScaleAbs(image, alpha=alpha/100, beta=-1*beta)
    #_, dst = cv2.threshold(i, threshold, 255, th_type)
    # cv2.destroyAllWindows()
    return R_mask, G_mask

if __name__=="__main__":
    
    # Image files location 
    location = 'Documents\GitHub\Optic_Disk\OCT_Manual_2'

    images, filenames = load_images_from_folder(location)
    
    global_threshold = 100

    wb = xlwt.Workbook()
    sheet_name = 'Threshold = ' + str(100)
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Image')
    sheet1.write(0, 1, 'Image J Marking')
    print('\nInitializing...')
    
    path = 'Documents\GitHub\Optic_Disk\OCT_Results\\'
    save = 'Documents\GitHub\Optic_Disk\OCT_Results_Manual_2' +'.xls'
    
    index = 0
    # Loop through all the images
    for i in images:

        print('\n  Image-' + str(index+1) + ' Loaded')
        
        sheet1.write(index+1, 0, filenames[index])
        img1 = i.copy()
        img = img1.copy()
        image = img.copy()

        R, G = threshold_tb(img1, global_threshold, 0)

        height, width = img.shape[0:2]

        X_R = np.zeros((height*width,2))
        wpx_R = 0
            
        X_G = np.zeros((height*width,2))
        wpx_G = 0
                
        for b in range(width):
            for a in range(height):

                if R[a,b] > 200:
                    X_R[wpx_R,0] = a    #column 0 has x coord of pixels which are white
                    X_R[wpx_R,1] = b    #column 1 has y coord of pixels which are white
                    wpx_R = wpx_R + 1   #counter to know the index of row
                        
                if G[a,b] > 200:
                    X_G[wpx_G,0] = a
                    X_G[wpx_G,1] = b
                    wpx_G = wpx_G + 1
    
        # To delete the unnecessary rows
        X_R = np.delete(X_R, slice(wpx_R,height*width), 0)
        X_G = np.delete(X_G, slice(wpx_G,height*width), 0)

        m_R = m_G = c_R = c_R = 0

        if len(X_R[:,0]) != 0 and len(X_R[:,1]) != 0:
            X_R_train, X_R_test, y_R_train, y_R_test = train_test_split(X_R[:,1].reshape((-1,1)), X_R[:,0], test_size=0.2, random_state=1) 
            # create linear regression object 
            reg_R = linear_model.LinearRegression() 
            # train the model using the training sets 
            reg_R.fit(X_R_train, y_R_train)
            m_R = reg_R.coef_[0]
            c_R = reg_R.intercept_
            fail_R = 0

        if len(X_G[:,0]) != 0 and len(X_G[:,1]) != 0:
            X_G_train, X_G_test, y_G_train, y_G_test = train_test_split(X_G[:,1].reshape((-1,1)), X_G[:,0], test_size=0.2, random_state=1) 
            reg_G = linear_model.LinearRegression() 
            reg_G.fit(X_G_train, y_G_train)
            m_G = reg_G.coef_[0]
            c_R = reg_G.intercept_
            fail_G = 0
            
        y_R_1 = c_R + m_R*1
        y_R_2 = c_R + m_R*(width-1)

        y_G_1 = c_R + m_G*1
        y_G_2 = c_R + m_G*(width-1)
            
        theta = math.degrees(math.atan(abs((m_G - m_R)/ (1 + m_R*m_G))))
            
        if ~fail_R and ~fail_G:
            print('\tTheta for '+ filenames[index] + ' = ', theta)
            sheet1.write(index+1, 1, theta)
        elif ~fail_R and fail_G:
            print('\tRight Half Indeterminate')
            sheet1.write(index+1, 1, 'Right Half Indeterminate')
        elif fail_R and ~fail_G:
            print('\tLeft Half Indeterminate')
            sheet1.write(index+1, 1, 'Left Half Indeterminate')
        else:
            print('\tBoth Halves Indeterminate')
            sheet1.write(index+1, 1, 'Both Halves Indeterminate')
            
            # cv2.imshow(filenames[index], picture_r)
            # cv2.waitKey(0)

        index = index + 1
        try:
            wb.save(save)
        except:
            print('\n  Unable to log data, file may be open' )
        else:
            print('\n  Data logged into file \n')
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------')
    
    wb.save(save)
    print('Execution Complete')
    cv2.destroyAllWindows()
