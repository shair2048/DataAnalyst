import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px
# import plotly.graph_objects as go

# global target_variable
# global independent_variables
global data

target_variable = None
independent_variable = None
data = None
columns = None
btn_choose_csv_file = None

def choose_variable(key_prefix, columns):
    global target_variable, independent_variable, btn_choose_csv_file
    if columns is not None:
        target_variable = st.selectbox('Select target variable', columns, index=None, key=f'{key_prefix}_target')
        independent_variable = st.multiselect('Select independent variable', columns, key=f'{key_prefix}_independent')
        btn_choose_csv_file = st.button("Excution", key=f'{key_prefix}_btn')
    else:
        st.write("columns is None")

def choose_model(df):
    data = df
    columns = list(data.columns)
    linear_regression, logicstic_regression, knn = st.tabs(["Linear Regression", "Logicstic Regression", "KNN"])
    
    with linear_regression:
        choose_variable('linear_regression', columns)
        model = LinearRegression()

        if btn_choose_csv_file:
            if len(independent_variable) == 1:
                # st.scatter_chart(data[[independent_variable[0], target_variable]])
                
                X = data[independent_variable[0]].values.reshape(-1, 1)
                y = data[target_variable].values
                model.fit(X, y)
                
                fig, ax = plt.subplots()
                sb.regplot(x=independent_variable[0], y=target_variable, data=data, scatter_kws={'alpha':0.5}, ax=ax)
                st.pyplot(fig)
                
                st.write("Coefficient:", model.coef_)
                st.write("Intercept:", model.intercept_)
            elif len(independent_variable) == 2:
                # Tạo mô hình hồi quy tuyến tính
                X = data[independent_variable].values.reshape(-1, 2)
                y = data[target_variable].values
                model.fit(X, y)
                
                # Tính toán giá trị dự đoán từ mô hình
                xx, yy = np.meshgrid(np.linspace(data[independent_variable[0]].min(), data[independent_variable[0]].max(), 100),
                                    np.linspace(data[independent_variable[1]].min(), data[independent_variable[1]].max(), 100))
                zz = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
                zz = zz.reshape(xx.shape)
                
                # Vẽ biểu đồ 3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data[independent_variable[0]], data[independent_variable[1]], data[target_variable], c=data[target_variable], cmap='viridis')
                ax.plot_surface(xx, yy, zz, alpha=0.5)  # Vẽ đường hồi quy
                ax.set_xlabel(independent_variable[0])
                ax.set_ylabel(independent_variable[1])
                ax.set_zlabel(target_variable)
                
                # plt.show()
                st.pyplot(fig)
                
                st.write("Coefficient:", model.coef_)
                st.write("Intercept:", model.intercept_)
            else:
                st.warning("Target variable or Independent variable have not been selected.")
        
    with logicstic_regression:
        choose_variable('logicstic_regression', columns)
        # model = LogisticRegression()
        
        # if btn_choose_csv_file:
        #     # if len(independent_variable) == 2:
        #         X = df[independent_variable].values
        #         y = df[target_variable].values
        #         # Huấn luyện mô hình
        #         model.fit(X, y)

        #         # Vẽ biểu đồ
        #         plt.figure(figsize=(10, 6))

        #         # Vẽ các điểm dữ liệu
        #         plt.scatter(df[independent_variable[0]], df[independent_variable[1]], c=df[target_variable], cmap='viridis', s=50, edgecolors='k', label='Data points')

        #         # Vẽ ranh giới quyết định của mô hình
        #         x_min, x_max = df[independent_variable[0]].min() - 1, df[independent_variable[0]].max() + 1
        #         y_min, y_max = df[independent_variable[1]].min() - 1, df[independent_variable[1]].max() + 1
        #         xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        #         Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        #         plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='red', label='Decision boundary')

        #         plt.xlabel(independent_variable[0])
        #         plt.ylabel(independent_variable[1])
        #         plt.title('Logistic Regression Decision Boundary')
        #         plt.legend()
        #         plt.colorbar()

        #         # Hiển thị biểu đồ
        #         st.pyplot(plt)
        
        
        
    with knn:
        choose_variable('knn', columns)
        
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
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=3) # k=5
            knn.fit(X_train_scaled, y_train)

            y_pred = knn.predict(X_test_scaled)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sb.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            st.pyplot(plt)
            
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy: ', accuracy)