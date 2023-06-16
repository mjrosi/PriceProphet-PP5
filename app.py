import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body


app = MultiPage(app_name= "PriceProphet: Housing Price Predictor") # Create an instance of the app 


# load pages scripts
app.add_page("Quick Project Summary", page_summary_body)


app.run() # Run the  app