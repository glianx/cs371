import streamlit as st
from matplotlib.figure import Figure

fig = Figure()
ax = fig.add_subplot()
ax.plot((1,2,3),(1,4,9))
st.pyplot(fig)