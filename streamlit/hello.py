import streamlit as st
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter((1,2,3,4),(1,4,9,16))
st.pyplot(fig)