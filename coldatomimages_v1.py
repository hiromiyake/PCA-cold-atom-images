#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:04:38 2017

@author: hiro
"""

import numpy as np
from sklearn.decomposition import PCA
import copy
#import matplotlib.pyplot as plt

## VERSION 1

class coldAtomImages:
    def __init__(self, imagelist):
        self.imagelist = list(imagelist)
        self.imagelistcrop = list(imagelist)
        self.nimage = len(imagelist)
        self.xsize = imagelist[0][0].shape[1]
        self.ysize = imagelist[0][0].shape[0]
        self.xsizecrop = imagelist[0][0].shape[1]
        self.ysizecrop = imagelist[0][0].shape[0]
        
        ## Note these are flattened
        ## Should reshape if you want to plot them
        self.pwoa_mean = None
        self.pwoa_bases = None
        self.nbases = 0
        
        self.pcaimagelist = list(imagelist)
        
        self.maskimage = None
        
    def getImageList(self):
        return self.imagelist
    
    def getImageListCrop(self):
        return self.imagelistcrop
    
    def getNImage(self):
        return self.nimage
    
    def getImageSize(self):
        return (self.ysizecrop, self.xsizecrop)
    
    def getImageSizeCrop(self):
        return (self.ysizecrop, self.xsizecrop)
    
    def cropImages(self, crop):
        imagelistcrop = copy.deepcopy(self.imagelist)
        j = 0
        for i in imagelistcrop:
            k = 0
            for m in i:
                imagelistcrop[j][k] = m[crop[1][0]:crop[1][1],crop[0][0]:crop[0][1]]
                k += 1
            j += 1
        self.imagelistcrop = list(imagelistcrop)
        self.xsizecrop = self.imagelistcrop[0][0].shape[1]
        self.ysizecrop = self.imagelistcrop[0][0].shape[0]
        
    def showAtomsOriginal(self, iimage):
        absimg = np.true_divide(self.imagelist[iimage][0] - self.imagelist[iimage][2], self.imagelist[iimage][1] - self.imagelist[iimage][2])
        return absimg
        
    def showAtoms(self, iimage):
        absimg = np.true_divide(self.imagelistcrop[iimage][0] - self.imagelistcrop[iimage][2], self.imagelistcrop[iimage][1] - self.imagelistcrop[iimage][2])
        return absimg
    
    def showAtomsPCA(self, iimage):
        absimg = np.true_divide(self.pcaimagelist[iimage][0] - self.pcaimagelist[iimage][2], self.pcaimagelist[iimage][1] - self.pcaimagelist[iimage][2])
        return absimg
    
    def setPCAasDefault(self):
        self.imagelistcrop = list(self.pcaimagelist)
    
    def PWOABases(self, nbases = 1):
        self.nbases = nbases
        imagelist = copy.deepcopy(self.imagelistcrop)

        # Get the PWOA of the first image, which is the 2nd entry in the list
        # Turn it into a row vector/matrix
        pwoa_mat = imagelist[0][1].flatten()
        if self.nimage > 1:
            for i in imagelist[1:]:
                pwoa_mat = np.concatenate((pwoa_mat, i[1].flatten()), axis = 0)
                
        pwoa_mat = pwoa_mat.astype(float)
        self.pwoa_mean = np.mean(pwoa_mat,0)
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        pca = PCA(n_components = nbases)
        pca.fit(pwoa_mat)        
        self.pwoa_bases = pca.components_
        
    def showPWOABases(self, iimage):
        basis = np.reshape(self.pwoa_bases[iimage], (self.ysizecrop, self.xsizecrop)).copy()
        return basis
        
    def getPWOABases(self):
        return self.pwoa_bases
    
    def getPWOAMean(self):
        return np.reshape(self.pwoa_mean, (self.ysizecrop, self.xsizecrop)).copy()
    
    ## This function can be used for checking crop region with original image
    ## -mask specifies the masking region
    ##  [vertical_left vertical_right; horizontal_top horizontal_bottom]
    ##  [x value       x value       ; y value        y value          ]
    def showCropRegion(self, mask, iimage = 0):
        margin = 20
        borderval = 0.01
        #self.maskimage = self.imagelist[iimage][1].copy()
        ## Processed image
        self.maskimage = np.true_divide(self.imagelist[iimage][0] - self.imagelist[iimage][2], self.imagelist[iimage][1] - self.imagelist[iimage][2])
        #self.maskimage[self.maskimage < 0.01] = 0.01
        #self.maskimage[self.maskimage > 1] = 1
        ## Top line
        self.maskimage[mask[1][0]:(mask[1][0]+margin),mask[0][0]:mask[0][1]] = borderval
        ## Bottom line
        self.maskimage[(mask[1][1]-margin):mask[1][1],mask[0][0]:mask[0][1]] = borderval
        ## Left line
        self.maskimage[mask[1][0]:mask[1][1],mask[0][0]:(mask[0][0]+margin)] = borderval
        ## Right line
        self.maskimage[mask[1][0]:mask[1][1],(mask[0][1]-margin):mask[0][1]] = borderval
        ## Blacked out box
        #self.maskimage[mask[1][0]:mask[1][1],mask[0][0]:mask[0][1]] = 0
        
        return self.maskimage
    
    ## Note this function can be used for checking crop and mask region
    ## -mask specifies the masking region
    ##  [vertical_left vertical_right; horizontal_top horizontal_bottom]
    ##  [x value       x value       ; y value        y value          ]
    def showMaskRegion(self, mask, iimage = 0):
        margin = 20
        borderval = 0.01
        #self.maskimage = self.imagelist[iimage][1].copy()
        ## Processed image
        self.maskimage = np.true_divide(self.imagelistcrop[iimage][0] - self.imagelistcrop[iimage][2], self.imagelistcrop[iimage][1] - self.imagelistcrop[iimage][2])
        #self.maskimage[self.maskimage < 0.01] = 0.01
        #self.maskimage[self.maskimage > 1] = 1
        ## Top line
        self.maskimage[mask[1][0]:(mask[1][0]+margin),mask[0][0]:mask[0][1]] = borderval
        ## Bottom line
        self.maskimage[(mask[1][1]-margin):mask[1][1],mask[0][0]:mask[0][1]] = borderval
        ## Left line
        self.maskimage[mask[1][0]:mask[1][1],mask[0][0]:(mask[0][0]+margin)] = borderval
        ## Right line
        self.maskimage[mask[1][0]:mask[1][1],(mask[0][1]-margin):mask[0][1]] = borderval
        ## Blacked out box
        #self.maskimage[mask[1][0]:mask[1][1],mask[0][0]:mask[0][1]] = 0
        
        return self.maskimage
        
    def PCAImageList(self, mask, method = 0):
        j = 0
        for i in self.imagelistcrop:
            self.pcaimagelist[j] = self.PCAImage(i, mask, method)
            j += 1
    
    def getPCAImageList(self, mask):
        return self.pcaimagelist
    
    def PCAImage(self, image, mask, method = 0):
        image_new = copy.deepcopy(image)

        ## Zero the mask region and renormalize the bases
        pwoa_bases_zr = self.pwoa_bases.copy()
        j = 0
        for i in pwoa_bases_zr:
            basis = i.copy()
            basis = np.reshape(basis, (self.ysizecrop, self.xsizecrop))
            basis[mask[1][0]:mask[1][1],mask[0][0]:mask[0][1]] = 0
            basis = basis.flatten()
            #basis = basis/np.dot(basis,basis)
            basis = basis/np.sqrt(np.dot(basis,basis))
            pwoa_bases_zr[j] = basis.copy()
            j += 1
            
        ##------------
        ## Process PWA
        pwa = image_new[0].astype(float)
        pwa = pwa.flatten()
        pwa_ac = pwa - self.pwoa_mean
        pwa_ac = np.reshape(pwa_ac, (self.ysizecrop, self.xsizecrop)).copy()
        pwa_ac = pwa_ac.flatten()
        
        coeff_pwa = np.dot(pwa_ac, pwoa_bases_zr[0])
        for i in pwoa_bases_zr[1:]:
            coeff_pwa = np.concatenate((coeff_pwa, np.dot(pwa_ac, i)), axis = 0)
        
        ##-------------
        ## Process PWOA
        pwoa = image_new[1].astype(float)
        pwoa = pwoa.flatten()
        pwoa_ac = pwoa - self.pwoa_mean
        coeff_pwoa = np.dot(pwoa_ac, self.pwoa_bases[0])
        
        for i in self.pwoa_bases[1:]:
            coeff_pwoa = np.concatenate((coeff_pwoa, np.dot(pwoa_ac, i)), axis = 0)
        
        pwoa_new = pwoa.copy()
        for i in xrange(self.nbases):
            pwoa_new += coeff_pwa[i]*self.pwoa_bases[i] - coeff_pwoa[i]*self.pwoa_bases[i]
            
        if method == 0: ## Subtract from both PWA and PWOA
            pwa_new = pwa.copy()
            for i in xrange(self.nbases):
                pwa_new -= coeff_pwa[i]*self.pwoa_bases[i]
            image_new[0] = pwa_new
            image_new[0] = np.reshape(image_new[0], (self.ysizecrop, self.xsizecrop))
            pwoa_new = pwoa.copy()
            for i in xrange(self.nbases):
                pwoa_new -= coeff_pwoa[i]*self.pwoa_bases[i]
            image_new[1] = pwoa_new
            image_new[1] = np.reshape(image_new[1], (self.ysizecrop, self.xsizecrop))
        else: ## Subtract from only PWOA
            pwoa_new = pwoa.copy()
            for i in xrange(self.nbases):
                pwoa_new += coeff_pwa[i]*self.pwoa_bases[i] - coeff_pwoa[i]*self.pwoa_bases[i]
            image_new[1] = pwoa_new
            image_new[1] = np.reshape(image_new[1], (self.ysizecrop, self.xsizecrop))
        
        return image_new
        