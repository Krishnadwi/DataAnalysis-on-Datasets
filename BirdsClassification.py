import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import Bunch
import skimage as sk
import pandas_profiling
def load_image(Location ,dim=(128,128)):  
    
    image_dir = Path(Location)
    print(Path)
    folders = [directory for directory  in image_dir.iterdir() if directory.is_dir()]
    print(folders)
    categories  = [fo.name for fo in folders]
    print(categories)
    
    images = []
    flat_data =[]
    target = []
    
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = sk.io.imread(file)
            #print("Before Transformation ")
            #plt.imshow(img)
            #plt.show()
            
            img_resized = resize(img, dim, anti_aliasing=True, mode='reflect')
            #print("After Transformation")
            plt.imshow(img_resized)
            #plt.show()
            
            # List Method append I am using Hear
            
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
            
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    
    return Bunch(data = flat_data,
                target = target,
                target_names = categories,
                images = images)
Data = load_image("C:\\Users\\KIIT\\Downloads\\Birds")
Features = Data.data
target = Data.target
df = pd.DataFrame(Features)
df['Class']= target
df.to_csv("Birds_Classification.csv")
print(plt.imshow(Data.images[24]))
print(plt.imshow(Data.images[-24]))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Features,target,test_size=0.35)
from sklearn.neighbors import KNeighborsClassifier 
knn  = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
yhat = knn.predict(x_test)
print(yhat)
print(knn.predict_proba(x_test))
Train_Score = knn.score(x_train,y_train)
print(Train_Score)
Test_score = knn.score(x_test,yhat)
Comp = pd.DataFrame({"Actual_data":y_test,
             "New Predication":yhat})
print(Comp)