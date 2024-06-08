import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# import plotly.graph_objects as go

global target_variable
global independent_variables
global data

target_variable = None
independent_variable = None
data = None
columns = None
btn_choose_csv_file = None

linear_regression, logicstic_regression, knn = st.tabs(["Linear Regression", "Logicstic Regression", "KNN"])

with st.sidebar:
    file = st.file_uploader("Choose CSV file!", type="csv")

    if file is not None:
        file_content = file.getvalue()
        data = pd.read_csv(file)
        
        columns = list(data.columns)
        # target_variable = st.selectbox('Select target variable', columns)
        # independent_variable = st.multiselect('Select independent variable', columns)

    # btn_choose_csv_file = st.button("Excution")

def choose_variable(target_variable, independent_variable, btn_choose_csv_file):
    if columns is not None:
        target_variable = st.selectbox('Select target variable', columns)
        independent_variable = st.multiselect('Select independent variable', columns)
        btn_choose_csv_file = st.button("Excution")

with st.container():
    # if btn_choose_csv_file:
        # with dataframe_info:
        #     original_data = data
        #     st.dataframe(original_data)
            
        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         info_df = pd.DataFrame({
        #             "Data Type": data.dtypes,
        #             "NaN Count": data.isna().sum()
        #         })

        #         st.dataframe(info_df)

        #     with col2:
        #         btn_rename_col = st.button("Save Change")
        #         btn_clean_data = st.button("Clean Data")
        #         btn_change_data_type = st.button("Change Data Type")
        #         # btn_download_file = st.button("Download File Cleaned")
        #         st.download_button(
        #             label="Download File Cleaned",  
        #             data=file, 
        #             file_name="large_df.csv",
        #             mime="text/csv",
        #         )
                
        #         st.data_editor(data)
        
        # # if btn_rename_col:
        # #     st.data_editor(data)

        # if btn_clean_data:
        #     cleaned_data = original_data.dropna()
        #     st.dataframe(cleaned_data)
            
        
            
            # plot_data = data[independent_variable + [target_variable]]
    
            # # Tính tổng theo biến độc lập
            # grouped_data = plot_data.groupby(independent_variable)[target_variable].sum().reset_index()
            
            # # Vẽ biểu đồ tròn
            # fig = px.pie(grouped_data, names=target_variable, values=target_variable, title='Pie Chart')
            
            # # Hiển thị biểu đồ
            # st.plotly_chart(fig)
            
    with linear_regression:
        # choose_variable(target_variable, independent_variable, btn_choose_csv_file)

        if columns is not None:
            target_variable = st.selectbox('Select target variable', columns, index=None)
            independent_variable = st.multiselect('Select independent variable', columns)
            btn_choose_csv_file = st.button("Excution")

        # st.write(independent_variable, target_variable)
        
        if btn_choose_csv_file:
            if len(independent_variable) == 1:
                # st.scatter_chart(data[[independent_variable[0], target_variable]])
                
                model = LinearRegression()
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
                model = LinearRegression()
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
        st.header("Logicstic Regression")
        # choose_variable()
            
        # if btn_choose_csv_file:
        #     st.header("Logicstic Regression")
        
    with knn:
        st.header("KNN")
        # X_train, X_test, y_train, y_test = train_test_split(data, target_variable, test_size=0.2, random_state=42)
        
        # knn = KNeighborsClassifier(n_neighbors=3) # K=3

        # # Huấn luyện mô hình
        # knn.fit(X_train, y_train)
        
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_test_scaled = scaler.transform(X_test)  # Scaled test data
        # y_pred = model.predict(X_test_scaled)     # Predicted labels
        
        # # Tính toán ma trận nhầm lẫn (confusion matrix)
        # conf_matrix = confusion_matrix(y_test, y_pred)
        
        # # Vẽ heatmap
        # fig, ax = plt.subplots()
        # sb.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='d')
        # ax.set_xlabel('Predicted Label')
        # ax.set_ylabel('True Label')
        # ax.set_title('Confusion Matrix')
        # st.pyplot(fig)