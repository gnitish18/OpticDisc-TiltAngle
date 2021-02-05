#If you would like to know more, contact us at gudapatinitish9@gmail.com (G Nitish) or s.swedha.krmg@gmail.com (S Swedha).
#You may go to the master branch to see our work (check our abstract submitted to the conference on Ophthalmic Technologies XXXI, part of SPIE BiOS)
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

def threshold_tb(image, threshold = 60, th_type = 3):
    alpha = 100
    beta = 0
    s = 0
    '''
    cv2.namedWindow('Contrast')
    # create trackbars for color change
    cv2.createTrackbar('alpha','Contrast', alpha, 255, nothing)
    cv2.createTrackbar('beta','Contrast', beta, 255, nothing)
    cv2.createTrackbar('Threshold','Contrast', threshold, 255, nothing)
    cv2.createTrackbar('Type','Contrast', th_type, 4, nothing)
    # create switch for ON/OFF functionality
    cv2.createTrackbar('Switch', 'Contrast', s, 1, nothing)
    # get current positions of four trackbars
    while(s == 0):
        alpha = cv2.getTrackbarPos('alpha','Contrast')
        beta = cv2.getTrackbarPos('beta','Contrast')
        threshold = cv2.getTrackbarPos('Threshold','Contrast')
        th_type = cv2.getTrackbarPos('Type','Contrast')
        s = cv2.getTrackbarPos('Switch','Contrast')
        
        _, dst = cv2.threshold(image, 70, 255, 3)
        i = cv2.convertScaleAbs(dst, alpha=alpha/100, beta=-1*beta)
        cv2.imshow('i',i)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
        #i = clahe.apply(i)

        _, t = cv2.threshold(i, threshold, 255, 0)
        cv2.imshow('Threshold', t)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    '''
    i = cv2.convertScaleAbs(image, alpha=alpha/100, beta=-1*beta)
    _, dst = cv2.threshold(i, threshold, 255, th_type)
    # cv2.destroyAllWindows()
    return dst

def linear_regression(Xl,Xr):
    nl = Xl.shape[0]
    nr = Xr.shape[0]
  
    # mean of x and y vector 
    m_xl, m_yl = np.mean(Xl[:,0]), np.mean(Xl[:,1]) 
    m_xr, m_yr = np.mean(Xr[:,0]), np.mean(Xr[:,1]) 
  
    # calculating cross-deviation and deviation about Xl and Xr
    SS_xy_l = np.sum(Xl[:,1]*Xl[:,0]) - nl*m_yl*m_xl 
    SS_xx_l = np.sum(Xl[:,0]*Xl[:,0]) - nl*m_xl*m_xl 
    
    SS_xy_r = np.sum(Xr[:,1]*Xr[:,0]) - nr*m_yr*m_xr 
    SS_xx_r = np.sum(Xr[:,0]*Xr[:,0]) - nr*m_xr*m_xr 

    # calculating regression coefficients 
    bl = np.zeros((2))
    br = np.zeros((2))
    bl[1] = SS_xy_l / SS_xx_l
    bl[0] = m_yl - bl[1]*m_xl
    
    br[1] = SS_xy_r / SS_xx_r
    br[0] = m_yr - br[1]*m_xr

    return bl, br

def load_images_from_folder(folder):
    images = []
    filenames = []
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,fname))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_image is not None:
            images.append(gray_image)
            filenames.append(fname)
    return images, filenames

if __name__=="__main__":
    
    # Image files location 
    location = 'Documents\GitHub\Optic_Disk\OCT_Abstract'

    images, filenames = load_images_from_folder(location)
    
    global_threshold = 100

    wb = xlwt.Workbook()
    sheet_name = 'Threshold = ' + str(100)
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Image')
    sheet1.write(0, 1, 'No Erosion & Dilation')
    sheet1.write(0, 2, '1 Erosion, 2 Dilation')
    sheet1.write(0, 3, '2 Erosion, 3 Dilation')
    sheet1.write(0, 4, 'Image J Marking')
    print('\nInitializing...')
    
    path = 'Documents\GitHub\Optic_Disk\OCT_Abstract\\'
    save = 'Documents\GitHub\Optic_Disk\OCT_Abstract\OCT_CLAHE_Batch1' +'.xls'
    
    folder = 'Documents\GitHub\Optic_Disk\OCT_Abstract'
    try:
        os.makedirs(folder+'\\Case00')
        os.makedirs(folder+'\\Case12')
        os.makedirs(folder+'\\Case23')
        os.makedirs(folder+'\\Temp')
    except:
        print("Creation of directory failed")
    else:
        print("Directory created")
    
    index = 0
    # Loop through all the images
    for i in images:

        print('\n  Image-' + str(index+1) + ' Loaded')
        
        sheet1.write(index+1, 0, filenames[index])
        img1 = i.copy()
        img1 = cv2.GaussianBlur(img1,(5,5),0)
        
        img = img1.copy()
        image = img.copy()

        loc = path + 'Gaussian.jpg'
        cv2.imwrite(loc, img1)
        
        thresh = threshold_tb(img1, 60, 3)
        loc = path + 'th1.jpg'
        cv2.imwrite(loc, thresh)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))
        img1 = clahe.apply(thresh)

        loc = path + 'CLAHE.jpg'
        cv2.imwrite(loc, img1)

        thresh = threshold_tb(img1, 100, 0)
        loc = path + 'th2.jpg'
        cv2.imwrite(loc, thresh)

        height, width = img.shape[0:2]
        pivot = np.zeros((width,1))

        for b in range(width):
            for a in range(height):
                if thresh[a,b] > 200:
                    pivot[b] = a
                    break

        maximum = np.amax(pivot)
        for a in range(width):
            if pivot[a] == maximum:
                maxpos = a
                break

        partl = thresh.copy()
        partr = thresh.copy()

        print('\tPivot Determined')

        for b in range(width):
            fl = 0
            for a in range(height):
                if partl[a,b] > 200:
                    fl = fl + 1
                if partl[a,b] < 10 and fl < 20:
                    partl[a,b] = 255
                if fl > 20:
                    break
            fl = 0
            for a in range(height):
                if partl[a,b] < 10:
                    fl = fl + 1
                if partl[a,b] > 200 and fl < 20:
                    partl[a,b] = 0
                if fl > 20:
                    break
            fl = 0
            for a in range(height):
                if partl[a,b] > 200 and fl < 20:
                    fl = fl + 1
                    continue
                partl[a,b] = 0
        
        print('\tMasked the required band')

        partr = partl.copy()

        for a in range(width):
            if a > 0.6*maxpos:
                partl[:, a] = 0
            if a < 1.4*maxpos:
                partr[:, a] = 0

        partl_og = partl.copy()
        partr_og = partr.copy()

        case = 0

        while (case != None):

            print('\n     Running Case-' + str(case))
            
            fail_l = fail_r = 1

            partl = partl_og.copy()
            partr = partr_og.copy()

            Xl = np.zeros((height*width,2))
            wpx_l = 0
            
            Xr = np.zeros((height*width,2))
            wpx_r = 0
        
            if case == 12:    
                
                kernel = np.ones((5,5),np.uint8)
                erosion_l = cv2.erode(partl,kernel,iterations = 1)
                dilation_l = cv2.dilate(erosion_l,kernel,iterations = 2)
                partl = dilation_l
                
                erosion_r = cv2.erode(partr,kernel,iterations = 1)
                dilation_r = cv2.dilate(erosion_r,kernel,iterations = 2)
                partr = dilation_r
            
            elif case == 23:

                kernel = np.ones((5,5),np.uint8)
                erosion_l = cv2.erode(partl,kernel,iterations = 2)
                dilation_l = cv2.dilate(erosion_l,kernel,iterations = 3)
                partl = dilation_l
                
                erosion_r = cv2.erode(partr,kernel,iterations = 2)
                dilation_r = cv2.dilate(erosion_r,kernel,iterations = 3)
                partr = dilation_r
                
            for b in range(width):
                for a in range(height):

                    if partl[a,b] > 200:
                        Xl[wpx_l,0] = a     #column 0 has x coord of pixels which are white
                        Xl[wpx_l,1] = b     #column 1 has y coord of pixels which are white
                        wpx_l = wpx_l + 1   #counter to know the index of row
                        
                    if partr[a,b] > 200:
                        Xr[wpx_r,0] = a
                        Xr[wpx_r,1] = b
                        wpx_r = wpx_r + 1

            #print(wpx_l,wpx_r)

            # To delete the unnecessary rows
            Xl = np.delete(Xl, slice(wpx_l,height*width), 0)
            Xr = np.delete(Xr, slice(wpx_r,height*width), 0)
            
            #print(Xl.shape[0:2])
            #print(Xr.shape[0:2])

            ml = mr = cl = cr = 0        

            if len(Xl[:,0]) != 0 and len(Xl[:,1]) != 0:
                Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl[:,1].reshape((-1,1)), Xl[:,0], test_size=0.2, random_state=1) 
                # create linear regression object 
                reg_l = linear_model.LinearRegression() 
                # train the model using the training sets 
                reg_l.fit(Xl_train, yl_train) 
                ml = reg_l.coef_[0]
                cl = reg_l.intercept_
                fail_l = 0

            if len(Xr[:,0]) != 0 and len(Xr[:,1]) != 0:
                Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr[:,1].reshape((-1,1)), Xr[:,0], test_size=0.2, random_state=1) 
                reg_r = linear_model.LinearRegression() 
                reg_r.fit(Xr_train, yr_train)
                mr = reg_r.coef_[0]
                cr = reg_r.intercept_
                fail_r = 0
            
            yl_1 = cl + ml*1
            yl_2 = cl + ml*(width-1)

            yr_1 = cr + mr*1
            yr_2 = cr + mr*(width-1)

            post_picture = partl + partr
            picture = partl_og + partr_og
            picture_color = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

            if case == 0:
                picture_l = cv2.line(picture_color, (1,int(yl_1)),(int(width-1),int(yl_2)), (0,50,250),3)
                picture_r = cv2.line(picture_l, (1,int(yr_1)),(int(width-1),int(yr_2)), (0,150,250),3)
            elif case == 12:
                picture_l = cv2.line(picture_color, (1,int(yl_1)),(int(width-1),int(yl_2)), (250,100,0),3)
                picture_r = cv2.line(picture_l, (1,int(yr_1)),(int(width-1),int(yr_2)), (250,100,200),3)
            elif case == 23:
                picture_l = cv2.line(picture_color, (1,int(yl_1)),(int(width-1),int(yl_2)), (0,250,200),3)
                picture_r = cv2.line(picture_l, (1,int(yr_1)),(int(width-1),int(yr_2)), (0,200,50),3)
            
            theta = math.degrees(math.atan(abs((mr - ml)/ (1 + ml*mr))))
            
            if ~fail_l and ~fail_r:
                print('\tTheta for '+ filenames[index] + ' = ', theta)
                if case == 0:
                    sheet1.write(index+1, 1, theta)
                elif case == 12:
                    sheet1.write(index+1, 2, theta)
                elif case == 23:
                    sheet1.write(index+1, 3, theta)
            elif ~fail_l and fail_r:
                print('\tRight Half Indeterminate')
                if case == 0:
                    sheet1.write(index+1, 1, 'Right Half Indeterminate')
                elif case == 12:
                    sheet1.write(index+1, 2, 'Right Half Indeterminate')
                elif case == 23:
                    sheet1.write(index+1, 3, 'Right Half Indeterminate')
            elif fail_l and ~fail_r:
                print('\tLeft Half Indeterminate')
                if case == 0:
                    sheet1.write(index+1, 1, 'Left Half Indeterminate')
                elif case == 12:
                    sheet1.write(index+1, 2, 'Left Half Indeterminate')
                elif case == 23:
                    sheet1.write(index+1, 3, 'Left Half Indeterminate')
            else:
                print('\tBoth Halves Indeterminate')
                if case == 0:
                    sheet1.write(index+1, 1, 'Both Halves Indeterminate')
                elif case == 12:
                    sheet1.write(index+1, 2, 'Both Halves Indeterminate')
                elif case == 23:
                    sheet1.write(index+1, 3, 'Both Halves Indeterminate')
            
            # cv2.imshow(filenames[index], picture_r)
            # cv2.waitKey(0)

            path1 = path + 'Temp\\' + str(index+1) + '_case' + str(case) + '_pre_' + filenames[index]
            path2 = path + 'Temp\\' + str(index+1) + '_case' + str(case) + '_post_' + filenames[index]
            
            if case == 0:
                path3 = path + 'Case00\\' + str(index+1) + '_case' + str(case) + '_final_' + filenames[index]
            elif case == 12:
                path3 = path + 'Case12\\' + str(index+1) + '_case' + str(case) + '_final_' + filenames[index]
            elif case == 23:
                path3 = path + 'Case23\\' + str(index+1) + '_case' + str(case) + '_final_' + filenames[index]
            
            try:
                cv2.imwrite(path1, picture)
                cv2.imwrite(path2, post_picture)
                cv2.imwrite(path3, picture_r)
            except:
                raise Exception("\a\tCould not write image")
            else:
                print('\tFile-' + str(index+1) + ' under Case-' + str(case) + ' is Saved')

            if case == 0:
                case = 12
            elif case == 12:
                case = 23
            else:
                case = None

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
