import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import joblib
import zipfile
import requests
import io


url = "https://raw.githubusercontent.com/yousefsameh21/customer_churn_project/main/cleaned_df.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open("cleaned_df.csv"), index_col=0)

url = "https://raw.githubusercontent.com/username/customer_churn_project/main/Decision_Tree.pkl"
response = requests.get(url)
model = joblib.load(io.BytesIO(response.content))page=st.sidebar.radio('Pages',['Home','Uni-Variate Analysis', 'Bi-Variate Analysis', 'Multi-Variate Analysis','Model Prediction'])
if page=='Home':
    st.markdown("<h1 style='text-align: center; color: Silver; '>Customer Churn Project</h1>", unsafe_allow_html=True)
    st.image('https://media.licdn.com/dms/image/v2/D4D12AQEVyfblBXjyJQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1664222940349?e=2147483647&v=beta&t=3CyUZB2EBfKYgRLkoggeC3S3hjaHDOo9a54pZZHqLO0')
    st.header('Customer Churn Dataset')
    st.dataframe(df.head(10))
    st.title("Table of Variables")
    column_info = [
    {"Attribute": "age", "Description": "Customer's age in years."},
    {"Attribute": "gender", "Description": "Customer's gender (e.g., Male, Female, Other)."},
    {"Attribute": "tenure", "Description": "Number of months or years the customer has been with the company."},
    {"Attribute": "usage frequency", "Description": "How frequently the customer uses the service or product."},
    {"Attribute": "support calls", "Description": "Number of support or helpdesk calls made by the customer."},
    {"Attribute": "payment delay", "Description": "Average delay in payments (in days or billing cycles)."},
    {"Attribute": "subscription type", "Description": "Type or plan of subscription (e.g., Basic, Premium, Enterprise)."},
    {"Attribute": "contract length", "Description": "Length or duration of the customer's contract or subscription period."},
    {"Attribute": "total spend", "Description": "Total amount of money spent by the customer on the service."},
    {"Attribute": "last interaction", "Description": "Date or number of days since the customer's last interaction with the service."},
    {"Attribute": "complaints", "Description": "Number of official complaints filed by the customer."},
    {"Attribute": "churn", "Description": "Whether the customer has left the service (1 = churned, 0 = active)."}
]

    df_info = pd.DataFrame(column_info)
    st.dataframe(df_info, use_container_width=True)
    # ----- Uni-Variate -----
elif page =='Uni-Variate Analysis':
    st.title("📊 Uni-Variate Analysis")

        # ----- 1️⃣ اختيار العمود -----
    st.subheader("1️⃣ Choose a Column to Analyze")
    column = st.selectbox('Select a column', df.columns)

        # ----- 2️⃣ اختيار نوع الرسم -----
    st.subheader("2️⃣ Choose Visualization Chart Type")
    chart_type = st.selectbox('Select chart type', ['Histogram', 'Bar'])

        # ----- 3️⃣ التحليل والرسم -----
    st.subheader("📈 Visualization")

        # لو المستخدم اختار histogram
    if chart_type == 'Histogram':
            # تأكد أن العمود رقمي
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(
                data_frame=df,
                x=column,
                nbins=20,
                template='plotly_dark',
                title=f"Distribution of {column}"
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Histogram works best with numeric columns. Please select a numeric column.")

        # لو المستخدم اختار bar chart
    else:
        df_counts = df[column].value_counts().reset_index()
        df_counts.columns = [column, 'count']
        fig = px.bar(
             df_counts,
             x=column,
             y='count',
             template='plotly_dark',
             title=f"Bar Chart for {column}"
            )
        st.plotly_chart(fig, use_container_width=True)
            
        
    # ----- Bi-Variate -----
elif page == "Bi-Variate Analysis":
    st.title("📈 Bi-Variate Analysis")
        
    st.write("### 1) Does Gender affect the likelihood of churn.")
    st.plotly_chart(px.histogram(df, x='gender', color='churn', barmode='group', title='Gender vs Churn'))
    st.write("### 2) Which Subscription_Type has the highest churn rate?")
    st.plotly_chart(px.histogram(df, x='subscription_type', color='churn', barmode='group', title='Subscription Type vs Churn'))
    st.write("#### 3) Does higher Total_Spend associate with lower churn?")
    st.plotly_chart(px.box(df, x='churn', y='total_spend', color='churn', title='Total Spend vs Churn'))
    st.write("#### 4) Are customers on short-term contracts (Monthly) more likely to leave than Annual contract holders?")
    contract_churn = df.groupby('contract_length')['churn'].mean().reset_index()
    contract_churn['Churn %'] = round(contract_churn['churn']*100, 2)
    contract_churn
    fig = px.bar(
    contract_churn,
    x='contract_length',
    y='Churn %',
    text='Churn %',
    title='Churn Rate by Contract Length',
    labels={'Churn %': 'Churn Rate (%)', 'contract length': 'Contract Type'},
    color='contract_length',
    color_discrete_sequence=px.colors.qualitative.Set2
                                                    )

    fig.update_traces(textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("1️⃣ Choose Two Columns to Compare")
    col1 = st.selectbox("Select X-axis column", df.columns)
    col2 = st.selectbox("Select Y-axis column", df.columns, index=1)

    st.subheader("2️⃣ Choose Chart Type")
    chart_type = st.selectbox("Select visualization type", ['Scatter', 'Box', 'Bar'])

    st.subheader("📊 Visualization")

    if chart_type == 'Scatter':
            fig = px.scatter(
                df,
                x=col1,
                y=col2,
                template='plotly_dark',
                title=f"Scatter Plot: {col1} vs {col2}"
            )
            st.plotly_chart(fig, use_container_width=True)

    if pd.api.types.is_numeric_dtype(df[col2]):
        df_grouped = df.groupby(col1)[col2].mean().reset_index()
        fig = px.bar(
            df_grouped,
            x=col1,
            y=col2,
            template='plotly_dark',
            title=f"Bar Chart: Average {col2} by {col1}"
        )
    else:
        # لو القيم نصية، نعرض توزيع عدد الحالات لكل فئة في col1
        df_grouped = df.groupby([col1, col2]).size().reset_index(name='count')
        fig = px.bar(
            df_grouped,
            x=col1,
            y='count',
            color=col2,
            template='plotly_dark',
            title=f"Bar Chart: Count of {col2} by {col1}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----- Multi-Variate -----
elif page== 'Multi-Variate Analysis': 
    st.title("📉 Multi-Variate Analysis")
    st.write("### How does the combination of Gender and Subscription_Type influence churn?")    
    st.plotly_chart(px.histogram(df, x='subscription_type', color='churn', facet_col='gender', barmode='group',
                   title='Churn by Gender and Subscription Type'))
    st.write("### Is there an interaction between Age, Tenure, and Usage_Frequency regarding churn?")    
    st.plotly_chart(px.scatter(df, x='age',    y='usage_frequency', color='churn',size='tenure',facet_col='churn',title='Age vs Usage_Frequency with Tenure as size, split by Churn'))
    st.write("### How does Contract_Length combined with Subscription_Type affect churn?")    
    st.plotly_chart(px.histogram(df, x='subscription_type', color='churn', facet_col='contract_length', barmode='group',
                   title='Churn by Subscription Type and Contract Length'))
    st.write("#### What is the distribution of customer Age by Subscription Type and Contract Length, and how does it relate to Churn?")    
    st.plotly_chart(px.box(df, x='subscription_type', y='age', color='churn', facet_col='contract_length',
             title='Age Distribution by Subscription Type, Contract Length and Churn'))
else:
    st.title("📉 Model Prediction")
    age = st.number_input("Age", min_value=int(df.age.min()), max_value=int(df.age.max()), step= 1)
    gender = st.selectbox("Gender", df.gender.unique())
    tenure = st.number_input("Tenure (months with company)", min_value=float(df.tenure.min()), max_value=float(df.tenure.max()))
    subscription_type = st.selectbox("Subscription Type", df.subscription_type.unique())
    contract_length = st.selectbox("Contract Length", df.contract_length.unique())
    total_spend = st.number_input("Total Spend ($)", min_value=float(df.total_spend.min()), max_value=float(df.total_spend.max()))
    support_calls = st.number_input("Enter Number of Support Calls", min_value=int(df.support_calls.min()), max_value=int(df.support_calls.max()))
    last_interaction = st.number_input("Enter Number of last interaction", min_value=(df.last_interaction.min()), max_value=(df.last_interaction.max()))
    usage_frequency=st.number_input("Enter Number of usage frequency", min_value=int(df.usage_frequency.min()), max_value=int(df.usage_frequency.max()))
    payment_delay=st.number_input("Enter Number of payment delay'", min_value=float(df.payment_delay.min()), max_value=float(df.payment_delay.max()))
    # Generate one raw Dataframe
    new_data = pd.DataFrame(data= [[age,gender,tenure,subscription_type,contract_length,total_spend,support_calls,
                                    last_interaction,usage_frequency,payment_delay]],

                                     columns= df.columns.drop('churn'))

    if st.button('Predict'):

        result = model.predict(new_data)[0]

        if result == 0:
            st.success('🟢 Active Customer')
        else:
            st.error('🔴 Churn Customer')

    



