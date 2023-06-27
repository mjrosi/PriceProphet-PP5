import streamlit as st


def page_summary_body():
    """
    Displays contents of the project summary page
    """
    st.write("### Quick Project Summary")

    st.info(
        f"**Project Terms & Jargons**\n\n"
        f"* **Sales price** of a house refers to the current market price, in US dollars,\
         of a house with with various attributes.\n"
        f"* **Inherited house** is a house that the client inherited from grandparents.\n"
        f"* **Summed price** is the total of the predicted sales prices of the four inherited houses.\n\n"
    )

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Dataset**\n"
        f"* The project dataset comes from housing price database from Ames, Iowa.\
         It is available in [Kaggle via Code Institute](https://www.kaggle.com/codeinstitute/housing-prices-data),\
         The dataset, which includes 1461 rows, represents housing records from Ames, Iowa. The dataset features 24 attributes indicative of the house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sales price for houses constructed between 1872 and 2010."
    )

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"Project [README](https://github.com/mjrosi/PriceProphet-PP5/tree/main#8-deployment) file.")

    # copied from README file - "Business Requirements" section
    st.success(
        f"**Project Business Requirements**\n\n"

        f"The project has 2 business requirements:\n"
        f"* **BR1** - The client wants to uncover how house attributes correlate with the sales price. Consequently, the client expects data visualizations of the variables correlated with the sales price.\n\n"
        f"* **BR2** - The client is keen on predicting the house sales price for her 4 inherited houses, and any other house in Ames, Iowa."
    )
