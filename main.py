import streamlit as st
import pandas as pd
import statsmodels.api as sm
from PIL import Image
import zipfile
import os
from scipy import stats
st.set_page_config(layout="wide")
st.title('CGM-based regression model app (test)')
f=open('./Fig/1.txt', 'r')
st.write(f.read())
f.close()

st.write('This app accepts CGM data in the following format. Missing values should be interpolated. In addition, data from the first day may be less reliable. Furthermore, the number of measurement days can affect AC_Mean and AC_Var, so it may be necessary to standardise the measurement period across participants.')

image = Image.open('./Fig/CGM_data.png')
st.image(image, caption='',use_column_width=True)

st.subheader('License')
f=open('./Fig/2.txt', 'r')
st.write(f.read())
f.close()

if(os.path.isfile('demo.zip')):
    os.remove('demo.zip')
with zipfile.ZipFile('demo.zip', 'x') as csv_zip:
    csv_zip.writestr("CGM_data.csv", 
                    pd.read_csv("CGM_data.csv").to_csv(index=False))        
with open("demo.zip", "rb") as file:
    st.download_button(label = "Download demo data",data = file,file_name = "demo.zip")

#Input
st.subheader('Upload CGM data')
df = st.file_uploader("", type="csv")

lagt=st.slider('Lag (used for AC_Mean and AC_Var calculation):', min_value=1, max_value=60, value=30, step=1)

if df is not None:
    df =pd.read_csv(df)
    AC= pd.DataFrame()
    for i in range (0,len(df.iloc[:,0])):
        X = df.iloc[i,2:]
        dff=pd.DataFrame(sm.tsa.stattools.acf(X,nlags=lagt,fft=False))
        AC=pd.concat([AC, pd.DataFrame([df.iloc[i,0],X.mean(),X.std(),dff.iloc[1:].mean()[0],dff.iloc[1:].var()[0]]).T])
    AC=AC.rename(columns={0: 'ID'}).rename(columns={1: 'Mean'}).rename(columns={2: 'Std'}).rename(columns={3: 'AC_Mean'}).rename(columns={4: 'AC_Var'})
    st.write(AC.set_index('ID'))
    
    st.write('Spearmans correlation between Mean and the objective value')
    st.write(stats.spearmanr(AC['Mean'],df.iloc[:,1]))
    st.write('Spearmans correlation between Std and the objective value')
    st.write(stats.spearmanr(AC['Std'],df.iloc[:,1]))
    st.write('Spearmans correlation between AC_Mean and the objective value')
    st.write(stats.spearmanr(AC['AC_Mean'],df.iloc[:,1]))
    st.write('Spearmans correlation between AC_Var and the objective value')
    st.write(stats.spearmanr(AC['AC_Var'],df.iloc[:,1]))
    
    options = ['Not perform a multiple rgression analysis',
               'Perform a multiple rgression analysis with AC_Var',
               'Perform a multiple rgression analysis with AC_Mean']
    selected_option = st.radio('Select AC_Var or AC_Mean to perform a multiple rgression analysis',options, key="1")
    if selected_option=='Not perform a multiple rgression analysis':
        if(os.path.isfile('CGM_regression.zip')):
            os.remove('CGM_regression.zip')
        with zipfile.ZipFile('CGM_regression.zip', 'x') as csv_zip:
            csv_zip.writestr("CGM_indices.csv",
                            AC.to_csv(index=False))
        with open("CGM_regression.zip", "rb") as file: 
            st.download_button(label = "Download the result",
                            data = file,file_name = "CGM_regression.zip")
    if selected_option=='Perform a multiple rgression analysis with AC_Var':
        AC=AC.reset_index(drop=True)
        Y=df.iloc[:,1]
        Full=AC[['Mean','Std','AC_Var']].apply(lambda x: (x-x.mean())/x.std(), axis=0)
        df_Full = sm.add_constant(Full)
        model = sm.OLS(Y, df_Full)
        result =model.fit()
        st.write("R2: "+str(result.rsquared))
        df_result=pd.concat([pd.DataFrame([result.params]).T.rename(columns={0: 'Coefficients'}),
                result.conf_int().rename(columns={0: '95%CI Lower'}).rename(columns={1: '95%CI Upper'})],axis=1)
        st.write(df_result)
        if(os.path.isfile('CGM_regression.zip')):
            os.remove('CGM_regression.zip')
        with zipfile.ZipFile('CGM_regression.zip', 'x') as csv_zip:
            csv_zip.writestr("CGM_indices.csv",
                            AC.to_csv(index=False))
            csv_zip.writestr("CGM_result.csv",
                            df_result.to_csv())
        with open("CGM_regression.zip", "rb") as file: 
            st.download_button(label = "Download the result",
                            data = file,file_name = "CGM_regression.zip")
    if selected_option=='Perform a multiple rgression analysis with AC_Mean':
        AC=AC.reset_index(drop=True)
        Y=df.iloc[:,1]
        Full=AC[['Mean','Std','AC_Mean']].apply(lambda x: (x-x.mean())/x.std(), axis=0)
        df_Full = sm.add_constant(Full)
        model = sm.OLS(Y, df_Full)
        result =model.fit()
        st.write("R2: "+str(result.rsquared))
        df_result=pd.concat([pd.DataFrame([result.params]).T.rename(columns={0: 'Coefficients'}),
                result.conf_int().rename(columns={0: '95%CI Lower'}).rename(columns={1: '95%CI Upper'})],axis=1)
        st.write(df_result)
        if(os.path.isfile('CGM_regression.zip')):
            os.remove('CGM_regression.zip')
        with zipfile.ZipFile('CGM_regression.zip', 'x') as csv_zip:
            csv_zip.writestr("CGM_indices.csv",
                            AC.to_csv(index=False))
            csv_zip.writestr("CGM_result.csv",
                            df_result.to_csv())
        with open("CGM_regression.zip", "rb") as file: 
            st.download_button(label = "Download the result",
                            data = file,file_name = "CGM_regression.zip")

    

