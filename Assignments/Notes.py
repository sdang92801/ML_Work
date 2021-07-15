

def Instan():

    print('''

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X=df.drop(columns='Drug')
    y=df['Drug']


    #instantiate
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3,stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    ''')

def Tranform():

    print('''
    #Transform Categorical Dataset
	#Ordinal Encoder for Ordinal Values
    
    import category_encoders as ce

    encoder = ce.OrdinalEncoder(cols=['score'],return_df=True,mapping=[{'col': 'score','mapping': {'Low': 0,'Medium' : 1,'High': 2}}])
    newDF = encoder.fit_transform(df)

    #df_dummies for Nomial Variables
    df_dummies = pd.get_dummies(newDF,columns=['instructor','course','semester'],drop_first=True)
    print(df_dummies.head().T)
    ''')

def AccuracyClassi():
    print('''
    # Measure Accuracy of Classification model:

    # #ROC Score:
    from sklearn.metrics import roc_auc_score, plot_roc_curve

    print(f'Training AUC: {roc_auc_score(y_train, logreg.predict_proba(X_train)[:,1])}')
    print(f'Testing AUC: {roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1])}')

    # plot_roc_curve(logreg, X_train, y_train)
    # plt.plot([0, 1], [0, 1], ls = '--', label = 'Baseline (AUC = 0.5)')
    # plt.legend()

    # #Specificity:

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, accuracy_score, recall_score

    y_pred=log_reg.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    print('Specificity: ',specificity)

    # #Precision:
    print('Precision: %.3f' % precision_score(y_test, y_pred))

    ''')

def AccuracyRegres():
    print('''
    # Measure Accuracy of Regression Model:

    from sklearn import metrics

    y_pred=clf.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    ''')

def EDA():
    print('''
    
    # EDA Options:

    # #Heatmap
    # plt.figure(figsize = (8, 5))
    # corr = df.corr()
    # mask = np.zeros_like(corr)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(corr, mask = mask, cmap = 'Blues', annot = True)
    # plt.show()

    # # #Histogram for Distribution
    # df.loc[:,['Drug','Na_to_K']].hist(bins=25,
    #                  figsize=(8,5),
    #                  xlabelsize='10',
    #                  ylabelsize='10',xrot=-15)
    # plt.show()

    # #Scatter Plot
    # sns.scatterplot(x='Age', y='Na_to_K', data=df, ci=None)
    # plt.show()

            sns.pairplot(fish,
             x_vars = ['Length1', 'Length2', 'Length3', 'Height', 'Width'],
             y_vars = ['Weight'],
             hue = 'Species')

    # fig, axes = plt.subplots(nrows = 1,ncols = 3,figsize = (8,2))
    # sns.regplot(x='Weight', y='Length1', data=df, ci=None, ax = axes[0], scatter_kws={'alpha':0.3})
    # sns.regplot(x='Weight', y='Length2', data=df, ci=None, ax = axes[1], scatter_kws={'alpha':0.3})
    # sns.regplot(x='Weight', y='Length3', data=df, ci=None, ax = axes[2], scatter_kws={'alpha':0.3})
    # fig.tight_layout()
    # plt.show()

    ''')

def KNN():
    print('''
    # KNN Both for Classification and Regressor:

    from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_train)
    score=knn.score(X_test,y_test)
    print('KNN Score: ',score)

    #Hypertuning - KNNClassifier
    ## Finding Optimal Lenght - KNN

    plt.plot(max_depth_length,accuracy)
    plt.show()

    max_depth_length = list(range(2,8))
    accuracy = []
    for depth in max_depth_length:
        clf=KNeighborsClassifier(n_neighbors=depth)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        accuracy.append(score)

    plt.plot(max_depth_length,accuracy)
    plt.show()

    #Hypertuning - KNeighborsRegressor
    ## Finding Optimal Lenght - KNN

    plt.plot(max_depth_length,accuracy)
    plt.show()

    max_depth_length = list(range(2,8))
    accuracy = []
    for depth in max_depth_length:
        clf=KNeighborsClassifier(n_neighbors=depth)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        accuracy.append(score)

    plt.plot(max_depth_length,accuracy)
    plt.show()

    ''')

def RandomForest():
    print('''
    # Random Forest:
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    clf = RandomForestClassifier(n_estimators=50,max_depth=4,min_samples_split=4,bootstrap=True,oob_score=True)
    clf.fit(X_train,y_train)
    clf.predict(X_test)
    score_train=clf.score(X_train,y_train)
    print('Random Forest Classifier Train: ',score_train)    
    score_test=clf.score(X_test,y_test)
    print('Random Forest Classifier Test: ',score_test)

    # # Randomized & Grid Search
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
    clf = RandomForestClassifier(random_state=3)

    param_grid={'n_estimators': [25,50,100,125,200],
           'min_samples_split': [5,6,7,8],
                'max_depth': [2,3,4,5]}

    cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
    cv_rs.fit(X_train,y_train)
    print('Random Search : ',cv_rs.best_params_)     

    cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
    cv_clf.fit(X_train,y_train)
    print('Best Parameter')
    print(cv_clf.best_params_)

    # # Randomized & Grid Search
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
    clf = RandomForestRegressor(random_state=3)

    param_grid={'n_estimators': [25,50,100,125,200],
           'min_samples_split': [5,6,7,8],
                'max_depth': [2,3,4,5]}

    cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
    cv_rs.fit(X_train,y_train)
    print('Random Search : ',cv_rs.best_params_)     

    cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
    cv_clf.fit(X_train,y_train)
    print('Best Parameter')
    print(cv_clf.best_params_)

    ''')

def DecisionTree():
    print('''
    # #Decision Tree for Classification and Regressor:

    from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

    clf = DecisionTreeClassifier(max_depth = 4, random_state = 3)
    clf.fit(X_train,y_train)
    clf.predict(X_test)
    print('Decision Tree Train: ',clf.score(X_train,y_train))
    print('Decision Tree Test: ',clf.score(X_test,y_test))

    ##Hyper Tuning
    ## Finding Optimal Depth - Decision Tree - Classifier

    max_depth_length = list(range(1,6))
    accuracy = []
    for depth in max_depth_length:
        clf=DecisionTreeClassifier(max_depth=depth,random_state=3)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        accuracy.append(score)
    
    # plt.plot(max_depth_length,accuracy)
    # plt.show()

    ## Finding Optimal Depth - Decision Tree - Regressor

    max_depth_length = list(range(1,6))
    accuracy = []
    for depth in max_depth_length:
        clf=DecisionTreeRegressor(max_depth=depth,random_state=3)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        accuracy.append(score)
    
    # plt.plot(max_depth_length,accuracy)
    # plt.show()

    ''')

def LogisticReg():
    print('''
    ## Logistic Regression - Classification
    # #Logistic Regression L1

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(C=1,penalty='l1', solver='liblinear',multi_class='ovr')
    log_reg.fit(X_train,y_train)
    print('Logistic Regression L1 Training Accuracy: ',log_reg.score(X_train,y_train))
    print('Logistic Regression L1 Testing Accuracy: ',log_reg.score(X_test,y_test))

    # #Logistic Regression L2

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(C=1,penalty='l2', solver='liblinear',multi_class='ovr')
    log_reg.fit(X_train,y_train)
    print('Logistic Regression L2 Training Accuracy: %.2f' % log_reg.score(X_train,y_train))
    print('Logistic Regression L2 Testing Accuracy: ',log_reg.score(X_test,y_test))
    
    # Assign_ML_Classification.py 
    # For Accuracy
    # Sensitivity
    # Specificity
    # AUC ROC
    # Additionally, plot the ROC.
    ''')

def Linear():
    print('''
    # #LinearRegression - For Regression
    from sklearn.linear_model import LinearRegression

    clf = LinearRegression(fit_intercept=True)
    clf.fit(X_train,y_train)
    clf.predict(X_test)
    score_train=clf.score(X_train,y_train)
    print('Linear Train: ',score_train)    
    score_test=clf.score(X_test,y_test)
    print('Linear Test: ',score_test)
    ''')

def Tuning():
    print('''
    # Hyper Tuning Parameters:
    #KNN and Decision Tree

    # Finding Optimal Depth - Decision Tree

    max_depth_length = list(range(1,6))
    accuracy = []
    for depth in max_depth_length:
        clf=DecisionTreeClassifier(max_depth=depth,random_state=3)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        accuracy.append(score)
    
    # plt.plot(max_depth_length,accuracy)
    # plt.show()


    # Finding Optimal Lenght - KNN

    plt.plot(max_depth_length,accuracy)
    plt.show()

    max_depth_length = list(range(2,8))
    accuracy = []
    for depth in max_depth_length:
        clf=KNeighborsClassifier(n_neighbors=depth)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        accuracy.append(score)

    plt.plot(max_depth_length,accuracy)
    plt.show()


    For Random Forest:

    # # Randomized & Grid Search
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
    clf = RandomForestClassifier(random_state=3)

    param_grid={'n_estimators': [25,50,100,125,200],
           'min_samples_split': [5,6,7,8],
                'max_depth': [2,3,4,5]}

    cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
    cv_rs.fit(X_train,y_train)
    print('Random Search : ',cv_rs.best_params_)     

    cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
    cv_clf.fit(X_train,y_train)
    print('Best Parameter')
    print(cv_clf.best_params_)

    ''')

def Kmean():
    print('''
    # KMeans & Scaling tools for Unsupervised learning :

    from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
    from sklearn.metrics import silhouette_score

    scaler=StandardScaler()
    scaled_df=scaler.fit_transform(df)

    #Silhouette Score:
    silhouette_scores=[]

    for i in range(2,12):
        kmean=KMeans(i)
        kmean.fit(scaled_df)
        silhouette_scores.append(silhouette_score(scaled_df,kmean.labels_))

    kmean=KMeans(n_clusters=2)
    kmean.fit(scaled_df)
    print('KMean Silhouette Score: ',silhouette_score(scaled_df, kmean.labels_))

    #Agglomerative Clustering:
    hc = AgglomerativeClustering(n_clusters = 2)
    hc.fit(scaled_df)
    print('Agglomerative Silhouette Score: ',silhouette_score(scaled_df, hc.labels_))

    #DBSCAN:	
    dbs = DBSCAN(eps = 0.5, min_samples = 3).fit(scaled_df)
    print('DBSCAN Silhouette Score: ',silhouette_score(scaled_df, dbs.labels_))
    # print(dbs.labels_)

    ''')
def Feature():
    print('''
    # Feature Importance:


    importance = clf.feature_importances_
    print(importance)
    plt.barh(X.columns,importance)
    plt.show()

    # Feature Importance (Doesnt work on Classification model)

    from sklearn.metrics import r2_score
    from rfpimp import permutation_importances

    def r2(rf, X_train, y_train):
        return r2_score(y_train, rf.predict(X_train))

    perm_imp_rfpimp = permutation_importances(clf, X_train, y_train, r2)

    print(perm_imp_rfpimp)
    x_values=list(range(len(perm_imp_rfpimp)))
    plt.bar(x_values,perm_imp_rfpimp['Importance'], orientation = 'vertical')
    plt.xticks(x_values,perm_imp_rfpimp.index,rotation='vertical')
    plt.title('Feature Importance')
    plt.show()
    ''')

def PCA():
    print('''
    from sklearn.decomposition import PCA
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3,stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)


    # from sklearn.decomposition import PCA
    # pca=PCA()
    # pca.fit(X_train)

    # plt.plot(range(1, 8), pca.explained_variance_ratio_[:7], marker = '.')
    # plt.xticks(ticks = range(1, 11))
    # plt.xlabel('Principal Component')
    # plt.ylabel('Proportion of Explained Variance')
    # plt.show()

    # pca = PCA(n_components = 6)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    ''')

def Pipeline():
    print('''
    from sklearn.pipeline import make_pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
    pipe = make_pipeline(StandardScaler(), PCA(n_components = 3), LogisticRegression())
    pipe.fit(X_train, y_train)
    print('Training accuracy:', pipe.score(X_train, y_train))
    print('Testing accuracy:', pipe.score(X_test, y_test))
    ''')

def GradientBoosting():
    print('''

    #Gradient Boosting Classifier
    #
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    print('Training accuracy Graident Boosting:', gbc.score(X_train, y_train))
    print('Testing accuracy Graident Boosting:', gbc.score(X_test, y_test))

    #Gradient Boosting Regression
    #
    #Gradient Boosting Regressor
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor()
    gbc.fit(X_train, y_train)
    print('Training accuracy Graident Boosting:', gbr.score(X_train, y_train))
    print('Testing accuracy Graident Boosting:', gbr.score(X_test, y_test))

    #LGBM Classifier

    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    print('Training accuracy LGBM:', lgbm.score(X_train, y_train))
    print('Testing accuracy LGBM:', lgbm.score(X_test, y_test))
    print('Predict')
    print(lgbm.predict(X_test))

    #LGBM Regressor
    
    from lightgbm import LGBMRegressor
    lgbm = LGBMRegressor()
    lgbm.fit(X_train, y_train)
    print('Training accuracy LGBM:', lgbm.score(X_train, y_train))
    print('Testing accuracy LGBM:', lgbm.score(X_test, y_test))
    print('Predict')
    print(lgbm.predict(X_test))

    # XGB Classifier

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    print('Training accuracy XGB:', xgb.score(X_train, y_train))
    print('Testing accuracy XGB:', xgb.score(X_test, y_test))

    # XGB XGBRegressor

    from xgboost import XGBRegressor
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    print('Training accuracy XGB:', xgb.score(X_train, y_train))
    print('Testing accuracy XGB:', xgb.score(X_test, y_test))
    ''')

def Neural():
    print('''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder,StandardScaler
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.python.keras.engine import sequential
    from tensorflow.keras.callbacks import EarlyStopping

    
    X=df.drop(columns='TARGET_5Yrs')
    y=df['TARGET_5Yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_shape=X_train.shape[1]

    model=Sequential()

    model.add(Dense(19,input_dim=input_shape,activation='relu'))

    model.add(Dense(10,input_dim=input_shape,activation='relu'))

    model.add(Dense(1, activation = 'sigmoid'))
    # For regression problems, we want to use the linear activation function.
    # For classification problems, we want to use the sigmoid activation function.

    model.compile(loss = 'bce', optimizer = 'adam')
    # For regression problems, we would want to use a loss function like MSE.
    # For binary classification problems like this one, we want to use the binary crossentropy loss. This can be abbreviated as "bce" in Keras.

    early_stopping = EarlyStopping(patience = 5)
    history = model.fit(X_train, y_train,
                        validation_data = (X_test, y_test), 
                        epochs=10, callbacks = [early_stopping])

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.legend()
    plt.show()
    ''')

def CNN():
    print('''
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras.utils import to_categorical

    from sklearn.model_selection import train_test_split

    df=pd.read_csv(r'C:\Users\dangs\Downloads\fashion-mnist_train.csv\fashion-mnist_train.csv')
    print(df.shape)

    X=df.drop(columns='label')
    y=df['label']

    X=X/255

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)

    X_train=X_train.to_numpy()
    X_test=X_test.to_numpy()

    print(X_train.shape)

    #If we had RBG (red, blue, green) values, we might reshape this to be 28x28x3
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    input_shape = X_train.shape[1:4]
    input_shape

    model = Sequential()

    model.add(Conv2D(filters = 8, # How many filters you want to use
                    kernel_size = 3, # size of each filter
                    input_shape = input_shape)) # What is the shape of your input features (we defined this above)
    # Pooling layer
    model.add(MaxPooling2D(pool_size = 2)) # Size of pooling
    # Flattening layer
    model.add(Flatten())
    # Output layer
    model.add(Dense(10, # How many output possibilities we have
                    activation = 'softmax'))

    # Step 2: Compile
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])


    # Step 3: Fit our model
    history = model.fit(X_train, y_train,
                        validation_data = (X_test, y_test), 
                        epochs=40)


    # Visualize the loss
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.legend()
    plt.show()

    # Visualize the accuracy
    plt.plot(history.history['acc'], label='Train Accuracy')
    plt.plot(history.history['val_acc'], label='Test Accuracy')
    plt.legend()
    plt.show()

    ''')

def RNN():
    print('''
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM

    df=pd.read_csv(r"C:\Users\dangs\Desktop\Python\ML_Work\Assignments\Files\AAPL.csv")

    plt.figure(figsize = (12, 5))
    df['Close'].plot()
    plt.ylabel('Closing Price')
    plt.show()

    train = df.loc[df['Date']<'2020-01-01', ['Close']]
    test = df.loc[df['Date']>='2020-01-01', ['Close']]

    print(train.shape)
    print(test.shape)

    scaler = MinMaxScaler(feature_range = (0, 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    train_features = TimeseriesGenerator(train, train, length = 5)

    # Step 1: Define our network structure
    # Save the input shape
    input_shape = train_features[0][0][0].shape
    input_shape

    # Sequential model
    model = Sequential()

    # LSTM layer
    model.add(LSTM(units = 50, 
                return_sequences = True, # True if next layer will be a recurrent layer
                input_shape = input_shape))
    model.add(LSTM(units = 50, 
                return_sequences = True))
    model.add(LSTM(units = 50, 
                return_sequences = False))
    # Output layer
    model.add(Dense(units = 1, activation = 'linear'))

    # Step 2: Compile
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    history = model.fit(train_features, 
                        epochs=100)

    plt.plot(history.history['loss'], label='Train loss')
    plt.legend()
    plt.show()

    test_features = TimeseriesGenerator(test, test, length = 5)

    preds = model.predict(test_features)

    plt.figure(figsize = (12, 5))
    plt.plot(scaler.inverse_transform(test), label='Actual Price')
    # Note: there are less prices for the predicted price than the actual price, 
    # because we need to use 5 days of prices to create our first prediction
    plt.plot(range(5, len(test)), scaler.inverse_transform(preds), label='Predicted Price')
    plt.title('Apple Closing Stock Price Prediction')
    plt.xlabel('Day (January 2020)')    
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    ''')
a='Yes'
while a=='Yes':
    q=input('As a Question?' \
            '\nInstantiate a Model: Instan'\
            '\nTransform Categorical Values: Transform'\
            '\nAccuracy of Classification Models: AccuracyClassi'\
            '\nAccuracy of Regression Models: AccuracyRegres'\
            '\nEDA: EDA'\
            '\nKNN Model: KNN'\
            '\nRandom Forest Model: RandomForest'\
            '\nDecision Tree Model: DecisionTree'\
            '\nLogistic Regression Model: LogisticReg'\
            '\nLinear Model: Linear'\
            '\nHyper Tuning Parameters: Tuning'\
            '\nUnsupervised KMeans and Tuning: Kmean'\
            '\nFeature Importance: Feature'\
            '\nPCA : PCA'\
            '\nPipeline : Pipeline'\
            '\ndef GradientBoosting(): GB'\
            '\nNeural : Neural'\
            '\nCNN : CNN'\
            '\nRNN : RNN'\
            '\nNo more questions: No\n'\
            )
    if q=="Instan":
        Instan()
    elif q=='Tranform':
        Tranform()
    elif q=='AccuracyClassi':
        AccuracyClassi()
    elif q=='AccuracyRegres':
        AccuracyRegres()
    elif q=='EDA':
        EDA()
    elif q=='KNN':
        KNN()
    elif q=='RandomForest':
        RandomForest()
    elif q=='DecisionTree':
        DecisionTree()
    elif q=='LogisticReg':
        LogisticReg()
    elif q=='Linear':
        Linear()
    elif q=='Tuning':
        Tuning()
    elif q=='Kmean':
        Kmean()
    elif q=='Feature':
        Feature()
    elif q=='PCA':
        PCA()
    elif q=='GB':
        GradientBoosting()
    elif q=='Neural':
        Neural()
    elif q=='CNN':
        CNN()
    elif q=='RNN':
        RNN()
    elif q=="No":
        a="No"
    else:
        print('Wrong Input')




    






