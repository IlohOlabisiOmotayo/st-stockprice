import streamlit as st
import sqlite3
import threading

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
    st.title("WELCOME TO STOCK PRICE PREDICTION APP")
    ACCOUNT = ["Login", "SignUp"]
    choice = st.selectbox("", ACCOUNT)

    if choice == 'Login':
        
        username = st.text_input("enter your username")
        password = st.text_input('password', type='password')
        
        if st.button("login"):
            create_usertable()
            result = login_user(username, password)
            if result:
                st.success("Logged in as {}".format(username))
                st.info("You can now access the Dashboard {}".format(username))
            else:
                st.warning("Incorrect username/password")

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

if __name__ == '__main__':
    bisi()