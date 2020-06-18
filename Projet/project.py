from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sys 
import math
import time
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

#Partie INTRODUCTION

def loadUsps(filename) :
    with open (filename ,"r" ) as f :
        f. readline ( )
        data =[ [float (x) for x in l.split () ] for l in f if len ( l.split () ) > 2]
    tmp = np.array (data)
    return tmp [:,1:] , tmp [:,0] .astype ( int )

def showUsps(data) :
    plt.imshow(data.reshape((16,16)), interpolation="nearest", cmap="gray")
    plt.colorbar()
    
def showWeights(weights,title=None) :
    plt.imshow(weights.reshape((16,16)))
    plt.colorbar()
    if title: plt.title(title)
    plt.show()
    
#only works for 2 labels !
def preProcessData(datax_train,datay_train,datax_test,datay_test,label1,label2):
    sub_datax_train = datax_train[(datay_train==label1) | (datay_train==label2)]
    sub_datay_train = datay_train[(datay_train==label1) | (datay_train==label2)]
    sub_datay_train=np.where(sub_datay_train==label1,1,-1)
    sub_datax_test = datax_test[(datay_test==label1) | ( datay_test==label2)]
    sub_datay_test = datay_test[(datay_test==label1) | ( datay_test==label2)]
    sub_datay_test=np.where(sub_datay_test==label1,1,-1)
    return sub_datax_train,sub_datay_train,sub_datax_test,sub_datay_test

def displayModelResult(model,name,datax_train,datay_train,datax_test,datay_test):
    print(name+" accuracy on train :",getAccuracy(datay_train,model.predict(datax_train)))
    print(name+" accuracy on test :",getAccuracy(datay_test,model.predict(datax_test)))
    showWeights(model.coef_,name+" : weights")
    print("Weight sum :",np.sum(np.abs(model.coef_)))
    
def getAccuracy(Y_true,Y_pred):
    class_pred=np.where(Y_pred>=0,1,-1)
    return (class_pred==Y_true).mean()

def preambule():
    print("\n###### PREAMBULE ######")
    #load data
    datax_train,datay_train = loadUsps ("./USPS/USPS_train.txt")
    datax_test,datay_test = loadUsps ("./USPS/USPS_test.txt")
    
    #preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(datax_train)
    datax_train = scaler.transform(datax_train)
    datax_test = scaler.transform(datax_test)
    
    accuracy_scorer = make_scorer(getAccuracy)
    alphas=[10000,1000,100,10,1,0.1,0.01,0.001]
    
    #45 car c'est le nombre total de cas à étudier (n(n-1))/2
    resultLinear = np.zeros((4,45))
    resultRidge = np.zeros((4,45))
    resultLasso = np.zeros((4,45))
    ClassIteration = 0
    n=10
    for i in range(10):
        for j in range(i+1,10) :
            print(ClassIteration)
            
            trainx,trainy,testx,testy=preProcessData(
                datax_train,datay_train,datax_test,datay_test,i,j)
        
            resultLinearTempo = np.zeros((4,n))
            resultRidgeTempo = np.zeros((4,n))
            resultLassoTempo = np.zeros((4,n))
            
            for t in range(n) :
                
                reg = linear_model.LinearRegression().fit(trainx,trainy)
                resultLinearTempo[0,t] = getAccuracy(trainy,reg.predict(trainx))
                resultLinearTempo[1,t] = getAccuracy(testy,reg.predict(testx))
                resultLinearTempo[2,t] = np.sum(np.abs(reg.coef_))
                resultLinearTempo[3,t] = np.sum(reg.coef_ == 0)
    
                ridgeCV=GridSearchCV(linear_model.Ridge(),{'alpha':alphas},cv=5,scoring=accuracy_scorer)
                ridgeCV.fit(trainx,trainy)
                resultRidgeTempo[0,t] = getAccuracy(trainy,ridgeCV.predict(trainx))
                resultRidgeTempo[1,t] = getAccuracy(testy,ridgeCV.predict(testx))
                resultRidgeTempo[2,t] = np.sum(np.abs(ridgeCV.best_estimator_.coef_))
                resultRidgeTempo[3,t] = np.sum(ridgeCV.best_estimator_.coef_ == 0)
                
                lassoCV=GridSearchCV(linear_model.Lasso(),{'alpha':alphas},cv=5,scoring=accuracy_scorer)
                lassoCV.fit(trainx, trainy)
                resultLassoTempo[0,t] = getAccuracy(trainy,lassoCV.predict(trainx))
                resultLassoTempo[1,t] = getAccuracy(testy,lassoCV.predict(testx))
                resultLassoTempo[2,t] = np.sum(np.abs(lassoCV.best_estimator_.coef_))
                resultLassoTempo[3,t] = np.sum(lassoCV.best_estimator_.coef_ == 0)
            
            resultLinear[:, ClassIteration] = np.mean(resultLinearTempo,axis=1)  
            resultRidge[:, ClassIteration] = np.mean(resultRidgeTempo,axis=1) 
            resultLasso[:, ClassIteration] = np.mean(resultLassoTempo,axis=1) 
            if (i == 0) and (j == 2) :
                print("\n###### Linear regression ######")
                displayModelResult(reg,"Linear regression",
                                     trainx,trainy,testx,testy)
                
                print("\n###### Ridge ######")
                ridge=ridgeCV.best_estimator_
                displayModelResult(ridge,"Ridge",
                                      trainx,trainy,testx,testy)
                
                print("\n###### Lasso ######")
                lasso=lassoCV.best_estimator_
                displayModelResult(lasso,"Lasso",
                                      trainx,trainy,testx,testy)

            ClassIteration += 1
    print("\n###### Linear regression ######")
    print(np.mean(resultLinear,axis = 1))
    
    print("\n###### Ridge ######")
    print(np.mean(resultRidge,axis = 1))
    
    print("\n###### Lasso ######")
    print(np.mean(resultLasso,axis = 1))


########----INPAINTING----########

def determineBestLasso(datax,datay) :
    datay = datay.reshape(-1,1)
    datax = datax.reshape(len(datay),-1)
    
    alphas=[10000,1000,100,10,1,0.1,0.01,0.001]

    lassoCV=GridSearchCV(linear_model.Lasso(),{'alpha':alphas},cv=5)

    lassoCV.fit(datax, datay)
    return lassoCV.best_params_,lassoCV.best_score_,lassoCV.best_estimator_

def rescale(data):
    return data/255
    
def read_im(fn):
    I=plt.imread(fn)
    I=rescale(I)
    return I

def fillImg(I,h):
    newI=np.zeros((int(math.ceil(np.shape(I)[0]/h)*h),
                  int(math.ceil(np.shape(I)[1]/h)*h),np.shape(I)[2]))-100
    newI[0:np.shape(I)[0],0:np.shape(I)[1]]=I
    return newI

def originalShape(I,shape):
    return I[0:shape[0],0:shape[1]]

def imshow(I, title=None, size=500, axis=False,final=True):
    """ display an image, with title, size, and axis """
    I=np.where(I<0,0,I)
    I=np.where(I>1,1,I)
    if final : plt.figure(figsize=(size//80, size//80))
    plt.imshow(I)
    if not axis: plt.axis('off')
    if title : plt.title(title)
    if final : plt.show()
    
def showImages(original,noisy,unnoised):
    """ Show the original, noisy and unnoised image in the same graph"""
    plt.figure(figsize=(9, 36))
    
    plt.subplot(131)
    imshow(original,"Original image",final=False)
    
    plt.subplot(132)
    imshow(noisy,"Noisy image",final=False)
    
    plt.subplot(133)
    imshow(unnoised,"Unnoised image",final=False)
    
    plt.show()
   
def getPatch(I,i,j,h):
    """
    

    Parameters
    ----------
    I : 
        The image.
    i : Integer
        the x coordonnate.
    j : Integer
        The y coordonnate.
    h : INteger 
        the lenght of the square.

    Returns
    -------
    (i,j) : Tuple of Integer
        Coordonnate of the patch.
    Patch : 
        The patch 

    """
    if(h%2==0):
        print("h must be an odd number")
        sys.exit("h must be an odd number")
    else:
        dist=int((h-1)/2)
        return (i,j),I[i-dist:i+dist+1,j-dist:j+dist+1]
    
def patchToVect(patch):
    """
    Tranform a patch to a vector

    """ 
    return np.array(np.concatenate(patch, axis=None))

def vectToPatch(vect):
    """ Tranform a vector to a patch """
    h=int(math.sqrt(np.shape(vect)[0]/3))
    return np.reshape(vect, (h,h,3))
    
def noise(I,prc,noisy_area=None):
    """
    Noise a Image

    Parameters
    ----------
    I : 
        Image.
    prc : float
        Probability of noise.
    noisy_area : TYPE, optional
        All coordonnate of the square to patch. The default is None.

    Returns
    -------
    I : TYPE
        Image with noise.

    """
    if(noisy_area):
        full_image_noise_prc=np.ones(np.shape(I)[:-1])
        noise_prc=np.random.random_sample((noisy_area[2],noisy_area[3]))
        full_image_noise_prc[noisy_area[0]:noisy_area[0]+noisy_area[2],
                             noisy_area[1]:noisy_area[1]+noisy_area[3]]=noise_prc
    else:
        full_image_noise_prc=np.random.random_sample(np.shape(I)[:-1])
    I=np.where(np.expand_dims(full_image_noise_prc,axis=2)>=prc,I,-100)
    return I
    
def deleteRect(I,i,j,height,width):
    """
    Delete a rectangle in the image

    Parameters
    ----------
    I : 
        Image.
    i : Integer
        The x coordonnate of the rectangle.
    j : Integer
        The y coordonnate of the rectangle.
    height : Integer
        The height of the rectangle.
    width : Integer
        The width of the rectangle..

    Returns
    -------
    J : 
        The image without the rectangle.

    """
    J = I.copy()
    J[i:i+height,j:j+width]=np.array([-100,-100,-100])
    return J
    
def getNoisyPatch(I,h,step): 
    """
    Get all noisy Patch

    Parameters
    ----------
    I : 
        Image.
    h : Integer
        The lenght of the noisy square.
    step : Integer
        DESCRIPTION.

    Returns
    -------
    noisy_patch : Dict
        All noisy patch with the index representing the coordonnate, and the 
        value representing the patch.

    """
    noisy_patch=dict()
    for i in range(0,I.shape[0]-h+1,step):
        for j in range(0,I.shape[1]-h+1,step):
            patch=getPatch(I,i+h//2,j+h//2,h)[1]
            if(-100 in patch):
                noisy_patch[i+h//2,j+h//2]=patch
    return noisy_patch

def getDictionnary(I,h,step): 
    """
    Get all good Patch

    Parameters
    ----------
    I : 
        Image.
    h : Integer
        The lenght of the noisy square.
    step : Integer
        DESCRIPTION.

    Returns
    -------
    dictionnary : Dict
        All good patch with the index representing the coordonnate, and the 
        value representing the patch.


    """
    dictionnary=dict()
    for i in range(0,I.shape[0]-h+1,step):
        for j in range(0,I.shape[1]-h+1,step):
            patch=getPatch(I,i+h//2,j+h//2,h)[1]
            if(not(-100 in patch)):
                dictionnary[i+h//2,j+h//2]=patch
    return dictionnary

def getAllPatch(I,h,step) :
    """
     Get all Patch (noisy and good)

    Parameters
    ----------
    I : 
        Image.
    h : Integer
        The lenght of the noisy square.
    step : Integer
        DESCRIPTION.

    Returns
    -------
    dict : Dict
        All patch with the index representing the coordonnate, and the 
        value representing the patch.

    """
    noisy_patch=getNoisyPatch(I,h,step)
    dictionnary=getDictionnary(I,h,step)
    return {"noisyPatch":noisy_patch,"goodPatch":dictionnary}


# All Function to reconstruct a noisy image
@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=DeprecationWarning)
def approximePatch(patchTarget,goodPatch, alpha = None) :
    """
    Appromixe a patch based on the good patch

    Parameters
    ----------
    patchTarget : 
        Patch to approxime.
    goodPatch : Dict
        Dict of good patch.
    alpha : TYPE, optional
        The alpha in the Lasso regression. If none, the function will search for the 
        the best alpha. The default is None.

    Returns
    -------
    Weight
        weight vector of the regression.
    Patch 
        return a tuple with the coordonnate of the patch and the patch

    """
    
    Y = patchToVect(patchTarget[1])

    boolGoodInfo = np.array(Y != -100)
        
    Y_train=Y[boolGoodInfo].reshape(-1,1)
    Y_test = Y[~boolGoodInfo].reshape(-1,1)

    X_train = np.zeros((len(Y_train),len(goodPatch)))
    X_test = np.zeros((len(Y_test),len(goodPatch)))

    i = 0
    for key,value in goodPatch.items() :
        X_train[:,i] = patchToVect(value)[boolGoodInfo]
        X_test[:,i] = patchToVect(value)[~boolGoodInfo]
        i += 1

    if alpha == None :
        param,score,Lasso = determineBestLasso(X_train, Y_train)
        alpha = param['alpha']
    
    else:
        Lasso = linear_model.Lasso(alpha = alpha)
        Lasso.fit(X_train,Y_train)
    
    w = Lasso.coef_
    Y[~boolGoodInfo] = Lasso.predict(X_test)
    
    return w, (patchTarget[0],vectToPatch(Y))

def reconstruction(patch,dico):
    """  Show the difference after the utilisation of approxime Patch"""
    
    w,Y_predic = approximePatch(patch,dico) 
    imshow(patch[1])
    imshow(Y_predic[1])
    
def newPatchInImage(I,patch) :
    """
    Replace the old patch by the new Patch

    Parameters
    ----------
    I : 
        Image.
    patch : Tuple 
        Tuple with the coordonante of the patch, and the patch.

    Returns
    -------
    I : 
        Image with the modification.

    """
    h= np.shape(patch[1])[0]
    dist = (int)((h-1)/2)
    I[patch[0][0]-dist:patch[0][0]+dist+1, patch[0][1]-dist:patch[0][1]+dist+1] = patch[1]
    return I

def reconstructAll(img,h,step) :
    """ Reconstruct all the image"""
    
    allPatch=getAllPatch(img,h,step)

    while len(allPatch["noisyPatch"])>0 :
    
        patchTarget=list(allPatch["noisyPatch"].items())[0]
        _,newPatch = approximePatch(patchTarget,allPatch["goodPatch"])
        img = newPatchInImage(img,newPatch)
        del allPatch["noisyPatch"][newPatch[0]]
    return img

# All Function to reconstruct an important area of the image 
def determineAllPixelToRepare(img):
    return np.argwhere(img[:,:,0]==-100)
        
def evaluatePatchNaive(patch):
    """ Naive Function to evaluate a patch"""
    return (patch[1][:,:,0]!=-100).sum()

def evaluatePatch(img,patch,confidenceValue):
    """ Second function to evaluate a patch """
    patchConfidenceValue=getPatch(confidenceValue,
                                  patch[0][0],patch[0][1],np.shape(patch[1])[0])[1]
    return np.mean(patchConfidenceValue)
    

def updateConfidenceValue(score,patch,confidenceValue):
    patchConfidenceValue=getPatch(confidenceValue,
                                  patch[0][0],patch[0][1],np.shape(patch[1])[0])[1]
    scorePatchValue=np.where(patch[1]==-100,score,patchConfidenceValue)
    newConfidenceValue=newPatchInImage(confidenceValue,(patch[0],scorePatchValue))
    return newConfidenceValue

def determineBestPatchToRepare(img,h,confidenceValue,method):
    pixelToRepare=determineAllPixelToRepare(img)
    bestScore=-1
    bestPatch=None
    for pixel in pixelToRepare:
        if pixel[0]<(h-1)/2:
            pixel=(int((h-1)/2),pixel[1])
        if pixel[1]<(h-1)/2:
            pixel=(pixel[0],int((h-1)/2))
        if pixel[0]>(np.shape(img)[0]-(h-1)/2)-1:
            pixel=(int((np.shape(img)[0]-(h-1)/2)-1),pixel[1])
        if pixel[1]>(np.shape(img)[1]-(h-1)/2)-1:
            pixel=(pixel[0],int((np.shape(img)[1]-(h-1)/2)-1))
        patch=getPatch(img,pixel[0],pixel[1],h)
        if method=="confidence":
            patchScore=evaluatePatch(img,patch,confidenceValue)
        else:
            patchScore=evaluatePatchNaive(patch)
        if(patchScore>bestScore):
            bestScore=patchScore
            bestPatch=patch
    #print("Best score : ",bestScore)
    return bestPatch,bestScore

def reconstructMissingPart(img,h,step,method="Naive"):
    """ Reconstruction of the missing part by using the function described above"""
    confidenceValue=np.where(img==-100,0,1).astype(float)
    allPatch=getAllPatch(img,h,step)
    while len(allPatch["noisyPatch"])>0 :
        patchTarget,scoreTarget=determineBestPatchToRepare(img,h,confidenceValue,method)
        _,newPatch = approximePatch(patchTarget,allPatch["goodPatch"])
        if method == "confidence" : confidenceValue=updateConfidenceValue(scoreTarget,patchTarget,confidenceValue)
        #imshow(confidenceValue)
        img = newPatchInImage(img,newPatch)
        #imshow(img)
        allPatch=getAllPatch(img,h,step)
    return img
    
def inpainting():
    print("\n###### INPAINTING ######")
    
    original_img=read_im("img.jpg")
    imshow(original_img)
    noisy_img=noise(original_img,0.3)
    imshow(noisy_img)
    noisy_img=noise(original_img,0.3,[70,90,50,50])
    imshow(noisy_img)
    noisy_img=deleteRect(original_img,40,60,50,50) 
    imshow(noisy_img)
    
    h=13
    t=time.time()
    noisy_img=noise(original_img,0.3,[70,90,50,50])
    filled_noisy_img=fillImg(noisy_img,h)
    denoised_img = reconstructAll(filled_noisy_img,h,h)
    final_denoised_img= originalShape(denoised_img,np.shape(original_img))
    showImages(original_img,noisy_img,final_denoised_img)
    print("Duration :",time.time()-t)
    
    t=time.time()
    noisy_img=deleteRect(original_img,40,60,50,50) 
    filled_noisy_img=fillImg(noisy_img,h)
    denoised_img = reconstructMissingPart(filled_noisy_img,h,h,"confidence")
    final_denoised_img= originalShape(denoised_img,np.shape(original_img))
    showImages(original_img,noisy_img,final_denoised_img)
    print("Duration :",time.time()-t)
    
if __name__ == "__main__":

    #preambule() 
    inpainting()


