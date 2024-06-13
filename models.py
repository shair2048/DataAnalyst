import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.datasets import make_moons
import graphviz
from matplotlib.colors import ListedColormap, to_rgb
# import plotly.graph_objects as go

global data

target_variable = None
independent_variable = None
data = None

def choose_variable(key_prefix, data):
    global target_variable, independent_variable, btn_choose_csv_file
    if data is not None:
        target_variable = st.selectbox('Select target variable', data.columns, index=None, key=f'{key_prefix}_target')
        independent_variable = st.multiselect('Select independent variable', data.columns, key=f'{key_prefix}_independent')
        btn_choose_csv_file = st.button("Execution", key=f'{key_prefix}_btn')
        le = LabelEncoder()

        if independent_variable:
            for column in independent_variable:
                if data[column].dtype == "object":
                    data[column] = le.fit_transform(data[column])
            if data[target_variable].dtype == "object":
                data[target_variable] = le.fit_transform(data[target_variable])
                
# def plot_decision_boundaries(X, y, classifier, title):
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
#                            np.arange(x2_min, x2_max, 0.01))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
#     plt.title(title)
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.show()

# X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
# def draw_meshgrid():
#     a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
#     b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

#     XX, YY = np.meshgrid(a, b)

#     input_array = np.array([XX.ravel(), YY.ravel()]).T

#     return XX, YY, input_array

def choose_model(df):
    global data, target_variable, independent_variable, btn_choose_csv_file
    data = df

    linear_regression, logistic_regression, knn, decision_tree, random_forest = st.tabs(["Linear Regression", "Logistic Regression", "KNN", "Decision Tree", "Random Forest"])
    
    with linear_regression:
        choose_variable('linear_regression', data)
        model = LinearRegression()

        if btn_choose_csv_file:
            if len(independent_variable) == 1:
                X = data[independent_variable[0]].values.reshape(-1, 1)
                y = data[target_variable].values
                model.fit(X, y)
                
                fig, ax = plt.subplots()
                sb.regplot(x=independent_variable[0], y=target_variable, data=data, scatter_kws={'alpha':0.5}, ax=ax)
                st.pyplot(fig)
                
                st.write("Coefficient:", model.coef_)
                st.write("Intercept:", model.intercept_)
            elif len(independent_variable) == 2:
                X = data[independent_variable].values.reshape(-1, 2)
                y = data[target_variable].values
                model.fit(X, y)
                
                xx, yy = np.meshgrid(np.linspace(data[independent_variable[0]].min(), data[independent_variable[0]].max(), 100),
                                    np.linspace(data[independent_variable[1]].min(), data[independent_variable[1]].max(), 100))
                zz = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
                zz = zz.reshape(xx.shape)
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data[independent_variable[0]], data[independent_variable[1]], data[target_variable], c=data[target_variable], cmap='viridis')
                ax.plot_surface(xx, yy, zz, alpha=0.5)
                ax.set_xlabel(independent_variable[0])
                ax.set_ylabel(independent_variable[1])
                ax.set_zlabel(target_variable)
                
                st.pyplot(fig)
                
                st.write("Coefficient:", model.coef_)
                st.write("Intercept:", model.intercept_)
            else:
                st.warning("Target variable or Independent variable have not been selected.")
        
    with logistic_regression:
        choose_variable('logistic_regression', data)
        model = LogisticRegression()

        if btn_choose_csv_file:
            X = data[independent_variable].values
            y = data[target_variable].values
            
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_train_transformed = label_encoder.fit_transform(y)
            elif y.dtype == 'float64':
                y_train_transformed = (y >= 0.5).astype(int)
            else:
                y_train_transformed = y.astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y_train_transformed, test_size=0.3, random_state=0)
            model = OneVsRestClassifier(LogisticRegression())
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            if len(np.unique(y_train)) == 2:  # Binary classification
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                st.pyplot(plt)
            else:  # Multiclass classification
                y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(y_test_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                plt.figure()
                colors = ['aqua', 'darkorange', 'cornflowerblue']
                for i, color in zip(range(y_test_bin.shape[1]), colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                             label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic for Multiclass')
                plt.legend(loc="lower right")
                st.pyplot(plt)

    with knn:
        choose_variable('knn', data)
        
        if btn_choose_csv_file:
            X = data[independent_variable].values
            y = data[target_variable].values
            
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_train_transformed = label_encoder.fit_transform(y)
            elif y.dtype == 'float64':
                y_train_transformed = (y >= 0.5).astype(int)
            else:
                y_train_transformed = y.astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_train_transformed, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train_scaled, y_train)

            y_pred = knn.predict(X_test_scaled)
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sb.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            st.pyplot(plt)
            
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy: ', accuracy)
            
    with decision_tree:
        choose_variable('decision_tree', data)
        
        # if btn_choose_csv_file:
        #     # Xác định các đặc trưng (features) và biến mục tiêu (target)
        #     X = data[independent_variable].values
        #     y = data[target_variable].values
            
        #     st.write(X)

        #     # Encode nhãn của biến mục tiêu nếu cần
        #     le = LabelEncoder()
        #     y = le.fit_transform(y)

        #     clf = tree.DecisionTreeClassifier(random_state=2021)
        #     clf = clf.fit(X, y)

        #     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[21, 6])

        #     colors = ['crimson', 'dodgerblue']
        #     ax1.plot(X[:, 0][y == 0], X[:, 1][y == 0], "o", color=colors[0])
        #     ax1.plot(X[:, 0][y == 1], X[:, 1][y == 1], "^", color=colors[1])
        #     xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
        #     pred = clf.predict(np.c_[(xx.ravel(), yy.ravel())])
        #     ax1.contourf(xx, yy, pred.reshape(xx.shape), cmap=ListedColormap(colors), alpha=0.25)

        #     # ax2.set_prop_cycle(mpl.cycler(color=colors)) # doesn't seem to work

        #     artists = tree.plot_tree(clf, feature_names=["X", "y"], class_names=colors,
        #                             filled=True, rounded=True, ax=ax2)
        #     for artist, impurity, value in zip(artists, clf.tree_.impurity, clf.tree_.value):
        #         # let the max value decide the color; whiten the color depending on impurity (gini)
        #         r, g, b = to_rgb(colors[np.argmax(value)])
        #         f = impurity * 2 # for N colors: f = impurity * N/(N-1) if N>1 else 0
        #         artist.get_bbox_patch().set_facecolor((f + (1-f)*r, f + (1-f)*g, f + (1-f)*b))
        #         artist.get_bbox_patch().set_edgecolor('black')

        #     plt.tight_layout()
        #     # plt.show()
        #     st.pyplot(plt)
        
        if btn_choose_csv_file:
            if len(independent_variable) == 1:
                X = data[independent_variable].values
                y = data[target_variable].values
                
                regressor = DecisionTreeRegressor(random_state = 0)  
    
                # fit the regressor with X and Y data 
                regressor.fit(X, y)
                
                X_grid = np.arange(min(X), max(X), 0.01) 
                X_grid = X_grid.reshape((len(X_grid), 1))  
                
                # scatter plot for original data 
                plt.scatter(X, y, color = 'red') 
                
                # plot predicted data 
                plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
                
                # specify title 
                plt.title('Decision Tree Regression')
                
                plt.xlabel(independent_variable[0])
                plt.ylabel(target_variable)
                # plt.export_graphviz(regressor, out_file ='tree.dot') 
                
                # show the plot
                st.pyplot(plt)
                
    with random_forest:
        choose_variable('random_forest', data)
        
        if btn_choose_csv_file: 
            
            X = data[independent_variable].values
            y = data[target_variable].values
            
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_train_transformed = label_encoder.fit_transform(y)
            elif y.dtype == 'float64':
                y_train_transformed = (y >= 0.5).astype(int)
            else:
                y_train_transformed = y.astype(int)
            
            SEED = 42
            X_train, X_test, y_train, y_test = train_test_split(X, y_train_transformed, test_size=0.2, random_state=SEED)

            rfc_ = RandomForestClassifier(n_estimators=100, 
                             max_depth=4,
                             random_state=SEED)
            rfc_.fit(X_train, y_train)
            y_pred = rfc_.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", accuracy)
            
            
            best_estimator_index = np.argmax(rfc_.feature_importances_)
            # Get the best estimator
            best_estimator = rfc_.estimators_[best_estimator_index]
            # Convert class names to strings
            class_names_str = [str(c) for c in np.unique(y)]
            # Plot the best decision tree
            fig, ax = plt.subplots(figsize=(12, 6))
            tree.plot_tree(best_estimator, feature_names=independent_variable, class_names=class_names_str, filled=True, ax=ax)
            st.pyplot(fig) 

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')

            # Display the plot directly in Streamlit
            st.pyplot(fig)
