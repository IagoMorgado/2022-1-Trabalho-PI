
import os
from math import pi
from time import time
import numpy as np
from PIL import Image
from PIL import ImageOps
from joblib import dump, load
from skimage.feature import greycoprops
from skimage.measure import shannon_entropy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def loadclf():
    return load('clf.joblib')


def saveclf(clf):
    dump(clf, 'clf.joblib')


def compute_entropy_for_glcm4d(glcm4d,distancesArr):
    entropyList = []
    for i in range(5):
            entropy = shannon_entropy(glcm4d[:, :, distancesArr[i], 0])
            entropyList.append(entropy)
    return entropyList

def ComputeMatrizCircular(x,y,xc,yc,imageArray,imgSize,distance,glcm4d):
    if (x+xc)<imgSize and (y+yc)<imgSize and (-y+yc)<imgSize and (-x+xc)<imgSize:
        glcm4d[imageArray[xc,yc],imageArray[(x+xc),(y+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(x+xc),(y+yc)],0,0]+1
        glcm4d[imageArray[xc,yc],imageArray[(x+xc),(-y+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(x+xc),(-y+yc)],0,0]+1
        glcm4d[imageArray[xc,yc],imageArray[(-x+xc),(y+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(-x+xc),(y+yc)],0,0]+1
        glcm4d[imageArray[xc,yc],imageArray[(-x+xc),(-y+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(-x+xc),(-y+yc)],0,0]+1
    
    if (y+xc)<imgSize and (x+yc)<imgSize and (-x+yc)<imgSize and (-y+xc)<imgSize:
        glcm4d[imageArray[xc,yc],imageArray[(y+xc),(x+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(y+xc),(x+yc)],0,0]+1
        glcm4d[imageArray[xc,yc],imageArray[(y+xc),(-x+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(y+xc),(-x+yc)],0,0]+1
        glcm4d[imageArray[xc,yc],imageArray[(-y+xc),(x+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(-y+xc),(x+yc)],0,0]+1
        glcm4d[imageArray[xc,yc],imageArray[(-y+xc),(-x+yc)],distance,0]=glcm4d[imageArray[xc,yc],imageArray[(-y+xc),(-x+yc)],0,0]+1
    
    return glcm4d 

def getMatrizCircular(imageArray,imgSize,glcm4d,distance):
    for i in range(imgSize):
        for j in range(imgSize):
            x=0
            y=distance
            p=3-(2*distance)
            glcm4d=ComputeMatrizCircular(x,y,i,j,imageArray,imgSize,distance,glcm4d)
        while x<y:
            if p<0: 
                p+=4*x+6
            else: 
                p+=4*(x-y)+10
                y=y-1
            x=x+1
            glcm4d=ComputeMatrizCircular(x,y,i,j,imageArray,imgSize,distance,glcm4d)
    return glcm4d

def compute_descriptors(image,greyNum):
    imgSize=image.size[0]
    imageArray = np.array(image, dtype=np.uint8)
    distancesArr=[1,2,4,8,16]
    glcm4d=[0]*(greyNum*greyNum*17*1)
    glcm4d=np.reshape(glcm4d,(greyNum,greyNum,17,1))
    for i in range(5):
        glcm4d=getMatrizCircular(imageArray, imgSize,glcm4d,distance=distancesArr[i])

    homogeneityMatrix = greycoprops(glcm4d, 'homogeneity')
    energyMatrix = greycoprops(glcm4d, 'energy')
    
    entropyList = compute_entropy_for_glcm4d(glcm4d,distancesArr)
    homogeneityList = np.hstack(homogeneityMatrix)
    energyList= np.hstack(energyMatrix)

    return list(energyList)+list(homogeneityList) + list(entropyList)

def processImageAndComputeDescriptors(path=None, image=None):
    if image is None:
        image = Image.open(path)
    imageEqualized = ImageOps.equalize(image)
    imageGray = imageEqualized.convert("L")
    image8Colors = imageGray.quantize(colors=8)
    image16Colors = imageGray.quantize(colors=16)
    image32Colors = imageGray.quantize(colors=32)
    allDescriptors = compute_descriptors(image32Colors,32) + compute_descriptors(image16Colors,16)+ compute_descriptors(image8Colors,8)
    return allDescriptors


def readImages(trainWindow):
    basePath = "imagens/"
    types = []
    imagesDescriptors = []
    num_of_images_processed = 0
    for i in range(1, 5):
        for entry in os.scandir(basePath + str(i) + "/"):
            if entry.path.endswith(".png") and entry.is_file():
                imagesDescriptors.append(processImageAndComputeDescriptors(entry.path))
                types.append(i)
                num_of_images_processed += 1
                trainWindow.progress['value'] = int((num_of_images_processed/400)*100)
                trainWindow.update_idletasks()
                trainWindow.labelVar.set(f"Gerando descritores {num_of_images_processed}/400")
                print(f"Gerando descritores {num_of_images_processed}/400")
    return imagesDescriptors, types


def trainclf(trainWindow):
    inicio = time()
    imagesDescriptors, types = readImages(trainWindow)
    trainWindow.progress.destroy()
    X_train, X_test, y_train, y_test = train_test_split(imagesDescriptors,
                                                        types,
                                                        test_size=.25)

    clf=MLPClassifier(solver='lbfgs', alpha=3.5,hidden_layer_sizes=(5, 2), random_state=1)
    print("Iniciando treinamento do Classificador")
    trainWindow.labelVar.set("Iniciando treinamento do Classificador...")
    clf.fit(X_train, y_train)
    infoString = ""
    y_predicted = clf.predict(X_test)
    print(f"Valores previstos pelo classificador {y_predicted}")
    print(f"Valores corretos esperados {y_test}")

    accuracy = accuracy_score(y_test, y_predicted)
    confusionMatrix = confusion_matrix(y_test, y_predicted)

    print(confusionMatrix)
    infoString += str(confusionMatrix)
    (mean_sensibility, specificity) = computeMetrics(confusionMatrix)
    fim = time()
    trainTime = fim-inicio
    print(f"Acurácia {accuracy}")
    infoString += f"\nAcurácia {accuracy}"
    infoString += f"\nSensiblidade Média: {mean_sensibility}"
    infoString += f"\nEspecificidade: {specificity}"
    infoString += f"\nTempo de treinamento  = {round(trainTime,2)} segundos"
    trainWindow.labelVar.set(infoString)

    print(f"Tempo de treinamento  = {round(trainTime, 2)} segundos")
    return clf


def computeMetrics(confusionMatrix):
    mean_sensibility = 0
    for i in range(0, 3):
        mean_sensibility += confusionMatrix[i][i] / 100

    sum = 0

    for i in range(0, 3):
        for j in range(0, 3):
            if i != j:
                sum += confusionMatrix[i][j] / 300
    specificity = 1 - sum
    print(f"Sensiblidade Média: {mean_sensibility}")
    print(f"Especificidade: {specificity}")
    return (mean_sensibility, specificity)
