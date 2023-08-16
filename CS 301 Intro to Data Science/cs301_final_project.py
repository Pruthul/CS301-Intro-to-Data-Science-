
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.cluster import KMeans

"""# **TASK 1**

Present a visual  distribution of the 3 classes. Is the data balanced? How do you plan to circumvent the data imbalance problem, if there is one? (hint: stratification needs to be included.) (1)
"""

data = pd.read_csv('fetal_health-1.csv')
data.head()

# Dropping duplicates columns from data
data.drop_duplicates(inplace=True)
#counting the data and plot
plot = sns.countplot(data=data, x='fetal_health', hue='fetal_health', palette=['r', 'g', 'b'])  
plt.show()

"""Data is imbalanced. As ploted in the graph above, fetal health=1 has higher count than other two.

# **TASK 2**

Present  10 features that are most reflective to fetal health conditions (there are more than one way of selecting features and any of these are acceptable) . Present if the correlation is statistically significant (using 95% and 90% critical values). (2)
"""

X = data.iloc[:, 0:21]  # independent columns
y = data.iloc[:, -1]    # target column (fetal health)

corr_data = data.select_dtypes(exclude="object")
corr_values = corr_data.corr()
#print the first 10 from the result 
corr_result = corr_values["fetal_health"].sort_values(ascending=False).head(10).to_frame()
print(corr_result)

"""# **Task** 3

Develop two different  models to classify CTG features into the three fetal health states (I intentionally did not name which two models. Note that this is a multiclass problem that can also be treated as regression, since the labels are numeric.) (2+2)
"""

print("********************************** MODEL 1 **********************************")
#Building the model for Random Forest Classifier 
scaler = StandardScaler().fit(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(scaler.transform(X), y, test_size=0.30, stratify=y,
                                                shuffle=True, random_state=21)

Xtrain, ytrain = RandomOverSampler(random_state=21).fit_resample(Xtrain, ytrain)
clf = RandomForestClassifier(random_state=21).fit(Xtrain, ytrain)
ypred_Model1 = clf.predict(Xtest)
print(classification_report(ytest, ypred_Model1))

print("********************************** MODEL 2 **********************************")
#Building the model for Decision Tree Classifier 
dtc = DecisionTreeClassifier()
ypred_Model2 = dtc.fit(Xtrain, ytrain).predict(Xtest)

print(classification_report(ytest, ypred_Model2))

"""# **TASK 4**

Visually present the confusion matrix (1)
"""

#Function for creating confusion matrix 
def plotConfusionMatrix(dtest, dpred, classes, title='Confusion Matrix',
                        width=0.75, cmap=plt.cm.Blues):
    cm = confusion_matrix(dtest, dpred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(np.shape(classes)[0] * width,
                                    np.shape(classes)[0] * width))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           aspect='equal')

    ax.set_ylabel('True', labelpad=20)
    ax.set_xlabel('Predicted', labelpad=20)

    plt.setp(ax.get_xticklabels(), rotation=90, ha='right',
             va='center', rotation_mode='anchor')

    fmt = '.2f'

    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.show()

#Random Forest classifier matrix all 3 claases 
print("Random Forest Confusion Matrix")
plotConfusionMatrix(ytest, ypred_Model1, classes=np.array(['Normal', 'Suspected', 'Pathological']), width=1.5, cmap=plt.cm.binary)
cm_rfm = confusion_matrix(ytest, ypred_Model1)
print(cm_rfm,"\n")
#Decision Tress classifier using all 3 classes 
print("Decision Tree Matrix")
plotConfusionMatrix(ytest, ypred_Model2, classes=np.array(['Normal', 'Suspected', 'Pathological']), width=1.5, cmap=plt.cm.binary)
cm_dct = confusion_matrix(ytest, ypred_Model2)
print(cm_dct)

"""# **TASK 5**

With a testing set of size of 30% of all available data, calculate (1.5)
Area under the ROC Curve
F1 Score
Area under the Precision-Recall Curve
(for both models in 3)
"""

#Model 1 - Random Forest 
print("********************************** MODEL 1 **********************************")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
Xtrain, ytrain = RandomOverSampler(random_state=21).fit_resample(Xtrain, ytrain)

# Instaniate the classification model and visualizer
model = RandomForestClassifier()
#ROC curve 
visualizer = ROCAUC(model, classes=["Normal", "Suspect", "Pathological"])
# Fit the training data to the visualizer
visualizer.fit(X_train, y_train) 
# Evaluate the model on the test data      
visualizer.score(X_test, y_test)   
# Finalize and print the figure    
visualizer.show()                       
#calculate F1 Score 
print("F1 Score:", f1_score(ytest, ypred_Model1, average='macro'))
#Creating Precesion Recall Curve 
viz = PrecisionRecallCurve(
    RandomForestClassifier(),
    classes=["Normal", "Suspect", "Pathological"],
    colors=["red", "green", "blue"],
    iso_f1_curves=True,
    per_class=True,
    micro=False
)
# Fit the training data to the visualizer
viz.fit(X_train, y_train)
# Evaluate the model on the test data 
viz.score(X_test, y_test)
# Finalize and print the figure 
viz.show()


print("********************************** MODEL 2 **********************************")


# Instaniate the classification model and visualizer
model = DecisionTreeClassifier()
# ROC curve
visualizer = ROCAUC(model, classes=["Normal", "Suspect", "Pathological"])
 # Fit the training data 
visualizer.fit(Xtrain, ytrain)  
# Evaluate the model on the test data     
visualizer.score(Xtest, ytest)  
# Finalize and print the figure    
visualizer.show()                       
#Calculating F1 Score 
print("F1 Score:", f1_score(ytest, ypred_Model2, average='macro'))

#Creating Precision Recall Curve 
viz = PrecisionRecallCurve(
    DecisionTreeClassifier(),
    classes=["Normal", "Suspect", "Pathological"],
    colors=["purple", "cyan", "blue"],
    iso_f1_curves=True,
    per_class=True,
    micro=False
)
# Fit the training data to the visualizer
viz.fit(Xtrain, ytrain)
# Evaluate the model on the test data 
viz.score(Xtest, ytest)
# Finalize and print the figure 
viz.show()

"""# **TASK 6**

Without considering the class label attribute, use k-means clustering to cluster the records in  different clusters and visualize them (use k to be 5, 10, 15). (2.5)
"""

X = data.iloc[:, 0:21].values

#K-means clustering for K=5
ktest = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = ktest.fit_predict(X)
print(pred_y)
plt.style.use('seaborn-whitegrid')
plt.scatter(X[:, 0], X[:, 1], c=pred_y, cmap=plt.cm.Paired)
plt.show()
#K-means clustering for K=5
ktest = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = ktest.fit_predict(X)
print(pred_y)
plt.scatter(X[:, 0], X[:, 1], c=pred_y, cmap=plt.cm.Paired)
plt.show()
#K-means clustering for K=5
ktest = KMeans(n_clusters=15, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = ktest.fit_predict(X)
print(pred_y)
plt.scatter(X[:, 0], X[:, 1], c=pred_y, cmap=plt.cm.Paired)
plt.show()