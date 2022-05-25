import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('training.csv')


X = df.drop(columns=['AQI'])
y = df['AQI']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = XGBRegressor(subsample=0.8,
                     n_estimators=1100,
                     min_child_weight=3,
                     max_depth=30,
                     learning_rate=0.05)


def predict(model, input):
    index = model.predict(input)
    print('Air Quality Index')
    return index


col = [['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']]


def main():
    st.header('Higgs Boson Event Detection')

    st.write('This is a simple demo of the Streamlit framework')

    st.subheader('Input the Data')
    st.write('Please input the data below')

    i = st.number_input('T',)
    j = st.number_input('TM',)
    k = st.number_input('Tm',)
    l = st.number_input('SLP',)
    m = st.number_input('H',)
    n = st.number_input('VV',)
    o = st.number_input('V',)
    p= st.number_input('VM')

    input = np.array([[i, j, k, l, m, n, o,p]])
  
    print(input)

    if st.button('Detect Event'):
        pred = predict(model, input)
        st.success('AQI is ' + pred)



if __name__ == '__main__':

    main()
