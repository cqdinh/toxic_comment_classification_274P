import streamlit as st
import numpy as np
from multiprocessing.connection import Client
from multiprocessing.context import AuthenticationError
import matplotlib.pyplot as plt

# Request a prediction from the model server
def predict(comment):
    address = ('localhost', 6000)
    with Client(address, authkey=b"274p_server") as conn:
        conn.send(comment)
        result = conn.recv()
    return result

st.title("Toxic Comment Classification")

# Widget to get text input
comment = st.text_input("Comment Text: ", "")

# Display the label predictions
try:
    # Get the probabilities
    prediction = predict(comment)

    labels = ("Toxic", "Severely Toxic", "Obscene", "Threatening", "Insulting", "Identity-based Hate")

    percents = prediction*100
    fig, ax = plt.subplots()

    # Display as bar graph
    yticks = list(range(6))
    ax.barh(yticks, percents, tick_label=labels, align="center")

    # Write the percentages at the end of the bars
    for i in range(6):
        ax.text(percents[i]+2, i, "{:.02f}%".format(percents[i]))

    # Display in the right order from top to bottom
    ax.invert_yaxis()

    # Alway sfrom 0-100%
    ax.set_xlim((0, 100))

    # Write percentages on x axis in 10% intervals
    xticks = list(range(0,101,10))
    ax.set_xticks(xticks)


    ax.set_xticklabels(["{:d}%".format(tick) for tick in xticks])

    # Show vertical grid lines
    ax.grid(True, axis="x")

    # Make the figure fit in the window
    st.pyplot(fig, False, bbox_inches='tight')

except AuthenticationError: # If server connection failed
    st.text("Connection Error")
