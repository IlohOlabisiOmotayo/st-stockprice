import streamlit as st
import sqlite3
import threading
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import yfinance as yf
import numpy as np
import joblib
import pickle
import streamlit_authenticator as stauth





# Create a thread-local storage for SQLite connection
sqlite_local = threading.local()

def get_sqlite_connection():
    if not hasattr(sqlite_local, 'connection'):
        sqlite_local.connection = sqlite3.connect('data.db')
    return sqlite_local.connection

def create_usertable():
    conn = get_sqlite_connection()
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    c.close()

def add_userdata(username, password):
    conn = get_sqlite_connection()
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    c.close()

def login_user(username, password):
    conn = get_sqlite_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username=? AND password=?', (username, password))
    data = c.fetchall()
    c.close()
    return data

def view_all_users():
    conn = get_sqlite_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    c.close()
    return data

def bisi():
    
    ACCOUNT = ["Login", "SignUp"]
    choice = st.selectbox("LOGIN/SIGNUP", ACCOUNT)

    if choice == 'Login':
        username = st.text_input("enter your username")
        password = st.text_input('password', type='password')

        if st.button("login"):
            create_usertable()
            result = login_user(username, password)
            if result:
                  st.success("Logged in as {}".format(username))
                  st.info("You can now access the Dashboard {}".format(username))
                    # Call the bbii() function when the login button is clicked
            else:
                st.warning("Incorrect username/password")
                return

    elif choice == "SignUp":
        st.subheader("Create new account")
        new_user = st.text_input("username")
        email = st.text_input("email")
        new_password = st.text_input("password", type='password')

        if st.button("sign up"):
            create_usertable()
            add_userdata(new_user, new_password)
            st.success("You have successfully created a valid account")
            st.info("Go to Login menu to login")


   

    

def analyser2(dt):
    """function creates a new column in the dataframe and helps to calculate polarity
    Input - Takes in the raw data
    Output - takes the output data
    """
    df = dt.copy()
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

def build_model2(dt, model):
    "Build the model for TESLA stock Price Prediction"
    df = dt.copy()
    # Splitting the dataset
    X = df[["sentiment"]]
    y = df["mean"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    if model == "rg":
        # Building the model
        model = make_pipeline(
            SimpleImputer(),
            Ridge()
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        intercept = model.named_steps["ridge"].intercept_
        coef = model.named_steps["ridge"].coef_
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        return model, rmse, intercept, coef, accuracy
    
    elif model == "lr":
        # Building the model
        model = make_pipeline(
            SimpleImputer(),
            LinearRegression(fit_intercept=True)
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        intercept = model.named_steps["linearregression"].intercept_
        coef = model.named_steps["linearregression"].coef_
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        return model, rmse, intercept, coef, accuracy

def make_prediction(df, model):
    # Perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["Tweet"].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Make predictions
    X = df[["sentiment"]]
    predictions = model.predict(X).round(2)

    return predictions.mean()

def build_model2(dt):
    "build the model"
    df = dt.copy()
    
    #splitting the dataset
    X = df[["sentiment"]]
    y = df["mean"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    
    #building the model
    model = make_pipeline(
        SimpleImputer(),
        Ridge()
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse= np.sqrt(mse)
    
    intercept = model.named_steps["ridge"].intercept_
    coef = model.named_steps["ridge"].coef_
    
    return model, rmse, intercept, coef

def execute_query(start_date, end_date):
    df = pd.read_csv('C://Users//USER//Desktop//multipages//pages//stock_2.csv', encoding='latin-1')

    df['Date'] = pd.to_datetime(df['Date'])
    conn = sqlite3.connect('tweetss.db')

    df.to_sql('tweeterss', conn, if_exists='replace', index=False)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    query = f"SELECT * FROM tweeterss WHERE Date >= '{start_date_str}' AND Date <= '{end_date_str}'"
    cursor = conn.execute(query)

    column_names = [description[0] for description in cursor.description]

    results = pd.DataFrame(cursor.fetchall(), columns=column_names)

    return results

def bbii():
    
   

    st.title("Stock Price Prediction")
    st.subheader("The stock data for Tesla (TSLA) within the specified date range will be retrieved.The retrieved data is stored in a DataFrame named stock_tweet, which contains information such as the date, opening price,closing price, and sentiment based on the tweet will be displayed for each trading day within the specified date range upon pressing the execute button and the predicted price based on the date range will be displayed upon pressing the predict button.")

    model= joblib.load("C://Users//USER//Desktop//multipages//pages//Stock_prices_model")

    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End Date')

    # Button to execute the code
    if st.sidebar.button('Execute'):
        results = execute_query(start_date, end_date)

        # Display the results
        st.write(results)

    if st.sidebar.button("Predict"):
        # Fetch data from the database for the specified dates
        results = execute_query(start_date, end_date)

        prediction = make_prediction(results, model)
        st.write(f"Predicted Stock price: ${prediction}")

    if st.sidebar.button("Sign Out"):
          st.info("You have successfully signed out")
              


def main():
    st.title("WELCOME TO STOCK PRICE PREDICTION APP")
    option = st.selectbox("WELCOME ", ("ACCOUNT", "DASHBOARD"))

    if option == "ACCOUNT":
        bisi()
    elif option == "DASHBOARD":
        bbii()



     
          

if __name__ == "__main__":
    main()


