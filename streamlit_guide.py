

"""
# My first app
Here's our first attempt at using data to create a table:
"""

# In python (both options work):
#streamlit run your_script.py [-- script args]
#python -m streamlit run your_script.py


import streamlit as st
import numpy as np
import pandas as pd


df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

#######################
df

#######################
st.write(df)

#######################
dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))


######################
st.text('Line chart')

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

######################
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

######################

st.text_input("Your name", key="name")


######################

st.title('You can call widgets that have a key, like the previous text input')

st.text(f'What you wrote before was: {st.session_state.name}')


##########################

st.markdown('### Use checkboxes to show/hide data')
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

##########################

st.markdown('### Use a selectbox for options')

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

