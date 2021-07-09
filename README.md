# Named-Entity-Recognition
NER with Tensorflow and Flask

### Step 1
Create Virtual Environment 

```virtualenv venv --python=python3```

```source venv/bin/activate  ```

### Step 2
Install Dependencies 

``` pip install -r requirements.txt ```

### Step 3
Train Model by going under Named-Entity_recognition/model/ and running following command

``` python model.py ```

### Step 4
Setup database, login to postgres and create a user

``` brew services start postgresql ```


``` psql postgres ```


``` CREATE ROLE dhrumilp WITH LOGIN PASSWORD ‘test123’; ```


``` CREATE DATABASE datapassports ```

Install pgAdmin4 and start a server with database name, username, and password at 127.0.0.1:5432

<img width="1203" alt="Screen Shot 2021-07-08 at 10 32 57 PM" src="https://user-images.githubusercontent.com/17984133/125021785-4d5b5e00-e049-11eb-93cc-9764f2f57923.png">

### Step 5
Run the flask app and go to http://localhost:5000/ you will be prompted to enter a link to scrape news article, provide any article link from https://www.aljazeera.com

``` cd WebApp ```

``` python app.py ```

<img width="332" alt="Screen Shot 2021-07-09 at 12 08 20 AM" src="https://user-images.githubusercontent.com/17984133/125022139-0326ac80-e04a-11eb-95fc-2abf013a67b3.png">


### Step 6
Check the results on http://localhost:5000/result with scrapped article and extracted name and organization

<img width="1440" alt="Screen Shot 2021-07-08 at 10 47 24 PM" src="https://user-images.githubusercontent.com/17984133/125022150-0883f700-e04a-11eb-823d-bb8313c10a6a.png">


