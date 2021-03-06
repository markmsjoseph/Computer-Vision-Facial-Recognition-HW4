import scipy.io as sio
import numpy as np
import re
from scipy import signal
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import cmath
import math
import time
import scipy.ndimage as ndimage
from PIL import Image
from numpy import linalg as LA

#=========================  WAVELET FUNCTIONS  ===================================

#define the morlet function that return the real part
def morlet_real(x, y, sig, theta, C1, C2):
    # set variables
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pie = np.pi
    # one peak morlet wave function with greek letter equal to 4
    exponentOfEInsideBrackets = (pie / (2 * sig)) * ((x * cosTheta) + (y * sinTheta))
    exponentOfEOutsideBrackets = -(x**2 + y**2)/ (2 * sig**2)

    #morlet wave function
    #cmath.rect(r, phi) Return the complex number x with polar coordinates r and phi.
    z = C1 / sig * (cmath.rect(1, exponentOfEInsideBrackets) - C2) * np.exp(exponentOfEOutsideBrackets)
    return z.real


#define the morlet function that return the imaginary part
def morlet_imag(x, y, sig, theta, C1, C2):
    # set variables
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pie = np.pi
    # one peak morlet wave function with greek letter equal to 4
    exponentOfEInsideBrackets = (pie / (2 * sig)) * ((x * cosTheta) + (y * sinTheta))
    exponentOfEOutsideBrackets = -(x**2 + y**2)/ (2 * sig**2)

    #morlet wave function
    #cmath.rect(r, phi) Return the complex number x with polar coordinates r and phi.
    z = C1 / sig * (cmath.rect(1, exponentOfEInsideBrackets) - C2) * np.exp(exponentOfEOutsideBrackets)
    return z.imag

def morlet_complex(x, y, sig, theta, C1, C2):
    # set variables
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pie = np.pi
    # one peak morlet wave function with greek letter equal to 4
    exponentOfEInsideBrackets = (pie / (2 * sig)) * ((x * cosTheta) + (y * sinTheta))
    exponentOfEOutsideBrackets = -(x**2 + y**2)/ (2 * sig**2)

    #morlet wave function
    #cmath.rect(r, phi) Return the complex number x with polar coordinates r and phi.
    z = C1 / sig * (cmath.rect(1, exponentOfEInsideBrackets) - C2) * np.exp(exponentOfEOutsideBrackets)

    return z

#finds the constants c2
def find_c2(xymin, xymax, sig, theta):
    numerator = 0
    denominator = 0
    cosine = np.cos
    cosineTheta = np.cos(theta)
    sineTheta = np.sin(theta)
    pie = np.pi
    for x in range(xymin, xymax+1, 1):
        for y in range( xymin, xymax+1, 1):
            numerator = numerator + (cosine((pie / (2 * sig)) * ((x * cosineTheta) + (y * sineTheta))) * np.exp(-(x**2 + y**2)/(2 * sig**2)))
            denominator = denominator + (np.exp(-(x**2 + y**2)/(2 * sig**2)))

    C2 = numerator/denominator
    return C2


#finds the constant c1
def find_c1(xymin, xymax, sig, theta, C2):
    Z = 0
    pie = np.pi
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    cosine = np.cos

    for x in range(xymin, xymax+1, 1):
        for y in range( xymin, xymax+1, 1):
            Z = Z + (1 - 2* C2 * cosine(pie/(2*sig) * ((x * cosTheta) + (y * sinTheta))) + C2**2) * np.exp((-(x**2 + y**2)/sig**2))
    C1 = 1/np.sqrt(Z)

    return C1


#plot the morlet function for the real
def morletMatrix_real(xymin, xymax, sig, theta):

    #find c1 and c2
    C2 = find_c2(xymin, xymax, sig, theta)
    C1 = find_c1(xymin, xymax, sig, theta, C2)

    #define grid over which the function should be plotted
    xx, yy = np.meshgrid(np.linspace(xymin, xymax, 33),np.linspace(xymin, xymax, 33))

    # fill a matrix with the morlet function values
    zz= np.zeros(xx.shape)
    for i in range(yy.shape[0]):
        for j in range(xx.shape[0]):
            zz[i,j] = morlet_real(xx[i,j], yy[i,j], sig, theta, C1, C2)

    return zz

# plot morlet function for imiginary
def morletMatrix_imag(xymin, xymax, sig, theta):
    #determine constants
    C2 = find_c2(xymin, xymax, sig, theta)
    C1 = find_c1(xymin, xymax, sig, theta, C2)

    #define grid over which the function should be plotted
    xx, yy = np.meshgrid(np.linspace(xymin, xymax, 33),np.linspace(xymin, xymax, 33))

    # fill a matrix with the morlet function values
    zz= np.zeros(xx.shape)
    for i in range(yy.shape[0]):
        for j in range(xx.shape[0]):
            zz[i,j] = morlet_imag(xx[i,j], yy[i,j], sig, theta, C1, C2)

    return zz

#morlet complex number
def morletMatrix_complex(xymin, xymax, sig, theta):
    #determine constants
    C2 = find_c2(xymin, xymax, sig, theta)
    C1 = find_c1(xymin, xymax, sig, theta, C2)

    #define grid over which the function should be plotted
    xx, yy = np.meshgrid(np.linspace(xymin, xymax, 33),np.linspace(xymin, xymax, 33))

    # fill a matrix with the morlet function values
    zz= np.zeros(xx.shape)
    for i in range(yy.shape[0]):
        for j in range(xx.shape[0]):
            zz[i,j] =morlet_complex(xx[i,j], yy[i,j], sig, theta, C1, C2)

    return zz

allvector = []# this will store all vectors needed to compute average face vector

#========================== LOAD FILE ==============================================


# get path


def do_everything(imageFileToOpen):
        print "\nLoading Image File"

        imageFile = imageFileToOpen
        im1 = Image.open(imageFile)

        width = 96
        height = 96

        #resize the image
        image = im1.resize((width, height), Image.BILINEAR)

        vector = []

        # used to calculate blocksizes
        def blockshaped(arr, nrows, ncols):

            h, w = arr.shape
            return (arr.reshape(h//nrows, nrows, -1, ncols)
                       .swapaxes(1,2)
                       .reshape(-1, nrows, ncols))

        #============================== LAYER 0 +===============================================

        print ("\nStarting Layer 0: Convolution with Gausian")

        # convolve with gausian
        gausianImageLayer0 = ndimage.gaussian_filter(image, sigma=(5, 5), order=0)
        # print gausianImageLayer0

        width = len(gausianImageLayer0)/4
        height = len(gausianImageLayer0[0])/4

        # divide gausian blur image into blocks have size 4x4 or row x col
        blocks = blockshaped(gausianImageLayer0, width, height)

        # find the max of each block
        for i in range(len(blocks)):
            layer0Max =  np.amax(blocks[i])
            vector.append(layer0Max)

        print str(imageFileToOpen) + " vector currently is " + str(vector)
        print "length of vector 0 is " + str(len(vector))
        print ("\n =================================================\n")




        #============================== LAYER 1 +===============================================



        print ("\nStarting Layer 1: ")
        start_time = time.clock()

        def level1(image, pie, sigma):

            print "Computing wavelet at angle " + str(pie) + " and sigma " + str(sigma)
            # compute real part
            kernel = morletMatrix_real(-16, 16, sigma, pie)
            realResponse = signal.convolve2d(image, kernel, boundary='symm', mode='same')
            # save real part for layer 2


            # compute imiginary part
            kernel = morletMatrix_imag(-16, 16, sigma, pie)
            imaginaryResponse = signal.convolve2d(image, kernel, boundary='symm', mode='same')
            #save imiginary response

            #get the magnitude of each pixel in the image
            magnitudeSigma3Theta0 = np.sqrt((realResponse**2)+(imaginaryResponse**2))

            # downsample. convolve each pixel(which is magnitude) with a gausian
            gausianSigma3Theta0 = ndimage.gaussian_filter(magnitudeSigma3Theta0, sigma=(5, 5), order=0)


            width = len(gausianSigma3Theta0)/4
            height = len(gausianSigma3Theta0[0])/4

            # divide gausian blur image into blocks have size 4x4 or row x col
            blocks = blockshaped(gausianSigma3Theta0, width, height)

            # find the max of each block
            for i in range(len(blocks)):
                layer0Max =  np.amax(blocks[i])
                vector.append(layer0Max)


        pies = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        for i in range(len(pies)):
            level1(image, pies[i], 3)
            level1(image, pies[i], 5)

        print "\nLayer 1 finished\n"
        print str(imageFileToOpen) + " vector currently is " + str(vector)
        print "length of vector 1 is " + str(len(vector))
        print "Time taken was " + str(time.clock() - start_time), "seconds\n"
        print ("\n =================================================\n")
        allvector.append(vector)




#=================================== AVERAGE FACE VECTOR  ====================================================

print ("\nStarting Average Vector Computation")
start_time = time.clock()


# open faces
for i in range(1, 2):
    for j in range(1, 3):
        image = "att_faces/s" + str(i) + "/" + str(j) + ".pgm"
        do_everything(image)

# # open non faces
# for i in range(1, 401):
#         image = "nonfaces/" + str(i) + ".png"
#         do_everything(image)


# summ all of the indicies at each index of each array
averageArray = np.zeros(len(allvector[0]))

for i in range(len(allvector)):
    for k in range(len(allvector[0])):
        averageArray[k] = averageArray[k] + allvector[i][k]
# print "max of all arrays"
# print averageArray



for j in range(len(averageArray)):
    averageArray[j] = averageArray[j]/len(allvector)
# print "average of all arrays"
# print averageArray


print "\nAverage Vactor finished\n"
print "length of vector Average Vector is " + str(len(averageArray))
print "Time taken was " + str(time.clock() - start_time), "seconds\n"
print ("\n =================================================\n")


#================================= Euclidean distance ==========================================
euclideanDistance = 0
arrOfArraysfaceVectorDifferenceFromAverage = [] # this will have at each index, an array. in that array is the values from the distance to the average

for i in range(len(allvector)):
    distanceFromAverage = []# this array will hold all 144 values and the value will be the distance from te average vector
    for k in range(len(allvector[0])):
        euclideanDistance = euclideanDistance + math.sqrt((averageArray[k] - allvector[i][k])**2)
        distanceFromAverage.append(euclideanDistance)
    arrOfArraysfaceVectorDifferenceFromAverage.append(distanceFromAverage)

# #debug
# print "length of the x matrix ie. the number of face vectors is " + str(len(arrOfArraysfaceVectorDifferenceFromAverage))
# print len(arrOfArraysfaceVectorDifferenceFromAverage[0])
# print arrOfArraysfaceVectorDifferenceFromAverage


#============================== STARTING COVARIANCE THING +===============================================


print "\nStarting Covariance Computation\n"


N = len(arrOfArraysfaceVectorDifferenceFromAverage)# get number of face vectors

covarianceMatrix =[[0]*144 for i in range(144)]#create an empty covariance matrix

for m in range(144):
    for n in range (144):
        summation = 0
        for z in range(N):
            sOfm = arrOfArraysfaceVectorDifferenceFromAverage[z][m]# get the m value
            sOfn = arrOfArraysfaceVectorDifferenceFromAverage[z][n]# get the n value
            product = np.dot(sOfm,sOfn)# get the dot product

            summation = summation + product# accumulate total

        answer = 1/(N-1) * summation

        covarianceMatrix[m][n] = answer

v = LA.eig(np.diag((1, 2, 3)))















#============================== LAYER 2 +===============================================

# print ("\nStarting Layer 2: ")
# start_time = time.clock()
#
# waveletResponses = []
# # compute wavelet response
# def computeWaveletComplexNumber(image, pie, sigma):
#     kernel = morletMatrix_complex(-16, 16, sigma, pie)
#     waveletResponse = signal.convolve2d(image, kernel, boundary='symm', mode='same')
#     return waveletResponse
#
#
# def level2(image, pie, sigma):
#
#     print "Computing wavelet at angle " + str(pie) + " and sigma " + str(sigma)
#     # compute real part
#     kernel = morletMatrix_real(-16, 16, sigma, pie)
#     realResponse = signal.convolve2d(image, kernel, boundary='symm', mode='same')
#
#
#     # compute imiginary part
#     kernel = morletMatrix_imag(-16, 16, sigma, pie)
#     imaginaryResponse = signal.convolve2d(image, kernel, boundary='symm', mode='same')
#
#
#     #get the magnitude of each pixel in the image
#     magnitudeSigma3Theta0 = np.sqrt((realResponse**2)+(imaginaryResponse**2))
#
#     # downsample. convolve each pixel(which is magnitude) with a gausian
#     gausianSigma3Theta0 = ndimage.gaussian_filter(magnitudeSigma3Theta0, sigma=(5, 5), order=0)
#
#     width = len(gausianSigma3Theta0)/4
#     height = len(gausianSigma3Theta0[0])/4
#
#     # divide gausian blur image into blocks have size 4x4 or row x col
#     blocks = blockshaped(gausianSigma3Theta0, width, height)
#
#     # find the max of each block
#     for i in range(len(blocks)):
#         layer0Max =  np.amax(blocks[i])
#         vector.append(layer0Max)
#
#
# # compute wavelet responses for level 1 stuff so we have an array with 8 'images'
# for i in range(len(pies)):
#     waveletResponses.append(computeWaveletComplexNumber(image, pies[i], 3))
#     waveletResponses.append(computeWaveletComplexNumber(image, pies[i], 5))
#
# # for each pie, compute the wavelet response for each of level 1 wavelet's responses
# for j in range(len(pies)):
#     for k in range(len(waveletResponses)):
#         level2(waveletResponses[k], pies[i], 3)
#         level2(waveletResponses[k], pies[i], 5)
#
#
# print "\nLayer 2 finished\n"
# print "vector currently is " + str(vector)
# print "length of vector is " + str(len(vector))
# print "Time taken was " + str(time.clock() - start_time), "seconds\n"
# print ("\n =================================================\n")







#====================================================== ANSWERS ===============================================
# faces
# Mean of all face vectors: 394517.830638
#
# Median of all face vectors: 385011.701466