import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from statsmodels.tools import add_constant

# Daten laden
sleep_productivity = pd.read_csv("/Users/feepieper/Library/CloudStorage/OneDrive-Persönlich/Ironhack/Module2/projects/project-data-analytics-and-preprocessing/data/sleep_cycle_productivity.csv")
sleep_health = pd.read_csv("/Users/feepieper/Library/CloudStorage/OneDrive-Persönlich/Ironhack/Module2/projects/project-data-analytics-and-preprocessing/data/Sleep_health_and_lifestyle_dataset.csv")
merged_df = pd.read_csv("/Users/feepieper/Library/CloudStorage/OneDrive-Persönlich/Ironhack/Module2/projects/project-data-analytics-and-preprocessing/data/merged_df.csv")
cleaned_df = pd.read_csv("/Users/feepieper/Library/CloudStorage/OneDrive-Persönlich/Ironhack/Module2/projects/project-data-analytics-and-preprocessing/data/cleaned_df.csv")

# Interaktionsspalte nur erstellen, falls noch nicht vorhanden (vermeidet Fehler bei Reload)
if 'Caffein_ScreenTime_Interaction' not in merged_df.columns:
    merged_df['Caffein_ScreenTime_Interaction'] = merged_df['Caffeine Intake (mg)'] * merged_df['Screen Time Before Bed (mins)']

st.title("Effects of and on Sleep Quality")

st.markdown("""
This project explores the relationships between various lifestyle and physiological factors and sleep quality.

The goal is to identify which variables significantly affect sleep quality and how sleep, in turn, impacts mood and productivity.

The analysis is hypothesis-driven and uses **Ordinary Least Squares (OLS)** regression models – including models with interaction terms – to test these relationships.
""")

tabs = st.tabs([
    "Data Preview", 
    "Hypotheses", 
    "EDA", 
    "Regression Plots", 
    "Correlation Matrix", 
    "OLS Regression",
    "Conclusion"
])

with tabs[0]:
    st.header("Data Preview")
    st.subheader("Sleep Productivity Dataset")
    st.dataframe(sleep_productivity.head())
    st.subheader("Sleep Health Dataset")
    st.dataframe(sleep_health.head())

with tabs[1]:
    st.header("Hypotheses")
    st.subheader("Effects on Sleep Quality")
    st.markdown("""
    **Q1: Does stress level influence sleep quality?**  
    - H₀: No relationship  
    - H₁: Higher stress levels negatively impact sleep quality
    """)
    st.markdown("""
    **Q2: Does BMI category affect sleep quality?**  
    - H₀: No effect  
    - H₁: Normal BMI correlates with better sleep
    """)
    st.markdown("""
    **Q3: Is resting heart rate related to sleep quality?**  
    - H₀: No relationship  
    - H₁: Lower resting HR predicts better sleep
    """)
    st.markdown("""
    **Q4: Do daily steps predict better sleep quality?**  
    - H₀: No effect  
    - H₁: More steps → better sleep
    """)
    st.subheader("Combined Effects on Sleep Quality")
    st.markdown("""
    **Q5: Interaction between age and exercise on sleep quality**  
    - H₀: No interaction  
    - H₁: Older people benefit more from exercise in terms of sleep quality
    """)
    st.markdown("""
    **Q6: Combined screen time and caffeine intake effect**  
    - H₀: No joint influence  
    - H₁: High values of both harm sleep quality
    """)
    st.subheader("Effects of Sleep Quality")
    st.markdown("""
    **Q7: Does sleep quality affect mood?**  
    - H₀: No effect  
    - H₁: Better sleep → better mood
    """)
    st.markdown("""
    **Q8: Does sleep quality affect productivity?**  
    - H₀: No effect  
    - H₁: Better sleep → higher productivity
    """)

with tabs[2]:
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Summary Statistics")
    st.dataframe(merged_df.describe())

    def plot_histogram_with_percentages(data, column, title):
        total = len(data)
        fig, ax = plt.subplots()
        sns.countplot(x=column, data=data, ax=ax)
        for p in ax.patches:
            count = p.get_height()
            percentage = f'{100 * count / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom')
        ax.set_title(title)
        st.pyplot(fig)

    plot_histogram_with_percentages(merged_df, 'Age Category', 'Histogram: Age Category (%)')
    
    fig, ax = plt.subplots()
    sns.boxplot(y="Age", data=merged_df, ax=ax)
    ax.set_title("Boxplot: Age")
    st.pyplot(fig)

    plot_histogram_with_percentages(merged_df, 'Gender', 'Histogram: Gender (%)')
    plot_histogram_with_percentages(merged_df, 'BMI Category', 'Histogram: BMI Category (%)')
    plot_histogram_with_percentages(merged_df, 'Stress Level', 'Histogram: Stress Level (%)')
    plot_histogram_with_percentages(merged_df, 'Sleep Quality', 'Histogram: Sleep Quality (%)')

    fig, ax = plt.subplots()
    sns.boxplot(y="Sleep Quality", data=merged_df, ax=ax)
    ax.set_title("Boxplot: Sleep Quality")
    st.pyplot(fig)

with tabs[3]:
    st.header("Regression Plots")

    st.subheader("Effects ON Sleep Quality")
    x_var = ['Age','Exercise (mins/day)', 'Stress Level','Caffeine Intake (mg)', 'Screen Time Before Bed (mins)']
    y_var = 'Sleep Quality'

    cols = st.columns(2)
    for i, x in enumerate(x_var):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(x=x, y=y_var, data=merged_df, ax=ax)
        ax.set_title(f'{y_var} vs {x}')
        cols[i % 2].pyplot(fig)

    st.subheader("Effects ON Sleep Quality for Heart Rate, Daily Steps and BMI")
    x_var_2 = ['Heart Rate', 'Daily Steps', 'BMI Category Code']
    fig, axs = plt.subplots(1, len(x_var_2), figsize=(5*len(x_var_2), 4))
    if len(x_var_2) == 1:
        axs = [axs]
    for i, x in enumerate(x_var_2):
        sns.regplot(x=x, y=y_var, data=cleaned_df, ax=axs[i])
        axs[i].set_title(f'Regplot: {y_var} vs {x}')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Effects OF Sleep Quality on Productivity and Mood")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(x='Sleep Quality', y='Productivity Score', data=merged_df, ax=axs[0])
    axs[0].set_title('Productivity Score vs Sleep Quality')
    sns.regplot(x='Sleep Quality', y='Mood Score', data=merged_df, ax=axs[1])
    axs[1].set_title('Mood Score vs Sleep Quality')
    plt.tight_layout()
    st.pyplot(fig)

with tabs[4]:
    st.header("Correlation Matrix")
    correlation_matrix = merged_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

with tabs[5]:
    st.header("OLS Regression")

    st.subheader("Model 1: Simple Effects on Sleep Quality (merged_df)")
    X = merged_df[['Stress Level', 'BMI Category Code', 'Heart Rate', 'Daily Steps']].copy()
    y = merged_df['Sleep Quality'].copy()
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    X_const = add_constant(X, has_constant='add')
    model1 = OLS(y, X_const).fit()
    st.text(model1.summary().as_text())

    st.subheader("Model 2: Simple Effects on Sleep Quality (cleaned_df)")
    X2 = cleaned_df[['BMI Category Code', 'Heart Rate', 'Daily Steps']].copy()
    y2 = cleaned_df['Sleep Quality'].copy()
    X2 = X2.fillna(X2.mean())
    y2 = y2.fillna(y2.mean())
    X2_const = add_constant(X2)
    model2 = OLS(y2, X2_const).fit()
    st.text(model2.summary().as_text())

    st.subheader("Model 3: Interaction Effect of Caffeine Intake and Screen Time on Sleep Quality")
    X3 = merged_df[['Caffeine Intake (mg)', 'Screen Time Before Bed (mins)', 'Caffein_ScreenTime_Interaction']].copy()
    y3 = merged_df['Sleep Quality'].copy()
    X3 = X3.fillna(X3.mean())
    y3 = y3.fillna(y3.mean())
    X3_const = add_constant(X3)
    model3 = OLS(y3, X3_const).fit()
    st.text(model3.summary().as_text())

    st.subheader("Model 4: Effect of Sleep Quality on Productivity Score")
    X4 = merged_df[['Sleep Quality']].copy()
    y4 = merged_df['Productivity Score'].copy()
    X4 = X4.fillna(X4.mean())
    y4 = y4.fillna(y4.mean())
    X4_const = add_constant(X4)
    model4 = OLS(y4, X4_const).fit()
    st.text(model4.summary().as_text())

    st.subheader("Model 5: Effect of Sleep Quality on Mood Score")
    X5 = merged_df[['Sleep Quality']].copy()
    y5 = merged_df['Mood Score'].copy()
    X5 = X5.fillna(X5.mean())
    y5 = y5.fillna(y5.mean())
    X5_const = add_constant(X5)
    model5 = OLS(y5, X5_const).fit()
    st.text(model5.summary().as_text())

with tabs[6]:
    st.header("Conclusion")

    st.markdown("""
    ### Interpretation of effects on sleep quality

    * R-squared: Only 0.07% of the target variable *Sleep Quality* is explained by the simple model, indicating low predictive power.
    * Coefficients show direction and strength of effects, but models have limited explanatory power.

    **Hypotheses from simple models:**
    1. Stress impacts sleep quality  
       * H₀ **not** rejected: p = 0.067 > 0.05 → no significant effect.
    2. BMI Category affects sleep quality  
       * H₀ **not** rejected: p = 0.854 > 0.05 → no significant effect.
    3. Heart rate impacts sleep quality  
       * H₀ **rejected**: p < 0.005 → higher heart rate associated with worse sleep.
    4. Daily steps affect sleep quality  
       * H₀ **not** rejected: p = 0.992 > 0.05 → no significant effect.
    
    **Hypotheses on interaction effects:**
                
    5. Age × Exercise on Sleep Quality  
       * H₀ **not** rejected: p = 0.236 > 0.05 → no significant interaction.
    6. Screen Time × Caffeine on Sleep Quality  
       * H₀ **not** rejected: p = 0.844 > 0.05 → no significant interaction.

    ---
    ### Interpretation of sleep quality effects on productivity and mood

    * How well does the model fit to predict *productivity* and *mood* by *sleep quality*?  
      R² values close to 0 (≈0.000) indicate no explanatory power.

    **Hypotheses:**
                
    7. Effect of Sleep Quality on Mood Score  
       * H₀ **not** rejected: p = 0.676 > 0.05 → no significant effect.
    8. Effect of Sleep Quality on Productivity Score  
       * H₀ **not** rejected: p = 0.934 > 0.05 → no significant effect.

    ___
    ### Overall Summary

    - Heart rate is a significant predictor of sleep quality.
    - No significant interaction effects were found for age × exercise or caffeine × screen time.
    - Sleep quality does **not** significantly predict mood or productivity in this dataset.
    - Overall low R² values suggest limited explanatory power of the models.

    ---
    This analysis highlights some significant predictors but also suggests that other factors beyond those studied here might better explain sleep quality and its downstream effects.
""")