import streamlit as st


def page_project_hypothesis_body():
    st.write("### Project Hypotheses and Validation")

    # conclusions taken from "03 - Data_Exploration" notebook
    st.success(
        f"**Hypothesis 1 - Space Impact:.** Larger houses, indicated by larger values for '2ndFlrSF', 'GarageArea', and 'TotalBsmtSF', are likely to be more expensive. These features represent the size of different areas of a house. The larger these areas, the higher the expected value of the house.\n\n"
        f"**Hypothesis 2 - Quality Influence:** The quality of a house's kitchen, denoted by 'KitchenQual', significantly impacts the house price. High-quality kitchens are a desirable feature for many homebuyers, and therefore, houses with high-quality kitchens are expected to have a higher sale price.\n\n"
        f"**Hypothesis 3 - Age and Renovation Factor:** The age and renovation status of a house, represented by 'YearBuilt' and 'YearRemodAdd', affect its value. Newly built or recently renovated houses are likely to fetch higher prices compared to older, unrenovated houses. If 'YearRemodAdd' is the same as 'YearBuilt', it means there have been no renovations, which might negatively impact the price.\n"
        f"* **Correct.** The Machine Learning model successfully validated all these hypotheses, indicating that these features indeed play a crucial role in determining a house's selling price.\n"
    )
