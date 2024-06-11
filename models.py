import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import plotly.express as px
import graphviz
# import plotly.graph_objects as go

# global target_variable
# global independent_variables
global data

target_variable = None
independent_variable = None
data = None
columns = None
values = None
btn_choose_csv_file = None

def choose_variable(key_prefix, values):
    global target_variable, independent_variable, btn_choose_csv_file
    if values is not None:
        target_variable = st.selectbox('Select target variable', values, index=None, key=f'{key_prefix}_target')
        independent_variable = st.multiselect('Select independent variable', values, key=f'{key_prefix}_independent')
        btn_choose_csv_file = st.button("Excution", key=f'{key_prefix}_btn')
    # else:
    #     st.write("values is None")

def choose_model(df):
    data = df
    # columns = list(data.columns)
    
    # label_encoder = LabelEncoder()
    # columns_transformed = label_encoder.fit_transform(data.columns)
    values = data.select_dtypes(include=['int64', 'float64']).columns
    # values = [col for col in values if col != 'id']
    linear_regression, logicstic_regression, knn, decision_tree = st.tabs(["Linear Regression", "Logicstic Regression", "KNN", "Decision Tree"])
    # data = data.apply(pd.to_numeric, errors='coerce')
    # info_df = pd.DataFrame({
    #                 "Data Type": data.dtypes,
    #                 "NaN Count": data.isna().sum()
    #             })
    # st.dataframe(columns_transformed, data.dtypes)
    # st.write(columns_transformed)
    
    with linear_regression:
        choose_variable('linear_regression', values)
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
        choose_variable('logicstic_regression', values)
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
            model.fit(X_train, y_train)

            # Dự đoán xác suất cho tập kiểm tra
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Tính toán đường cong ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Vẽ biểu đồ ROC
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
        
    with knn:
        choose_variable('knn', values)
        
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
            
    with decision_tree:
        choose_variable('decision_tree', values)
        
        # if btn_choose_csv_file:
        #     X = data[independent_variable].values
        #     y = data[target_variable].values
            
        #     if y.dtype == 'object':
        #         label_encoder = LabelEncoder()
        #         y_train_transformed = label_encoder.fit_transform(y)
        #     elif y.dtype == 'float64':
        #         y_train_transformed = (y >= 0.5).astype(int)
        #     else:
        #         y_train_transformed = y.astype(int)
            
        #     X_train, X_test, y_train, y_test = train_test_split(X, y_train_transformed, test_size=0.3, random_state=0)

        #     # Xây dựng mô hình Decision Tree và huấn luyện
        #     clf = DecisionTreeClassifier()
        #     clf.fit(X_train, y_train)

        #     # Dự đoán và đánh giá mô hình
        #     y_pred = clf.predict(X_test)
        #     accuracy = accuracy_score(y_test, y_pred)
        #     st.write('Accuracy:', accuracy)

        #     # Hiển thị cây quyết định (tùy chọn)
        #     st.subheader('Cây quyết định:')
        #     dot_data = export_graphviz(clf, out_file=None, feature_names=independent_variable, class_names=clf.classes_, filled=True, rounded=True, special_characters=True)
        #     graph = graphviz.Source(dot_data)
        #     st.graphviz_chart(graph)
