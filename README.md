# Final_year Project_2025
This is repo for PFA 2025

## How to run the project
1. Clone the repository
2. Install the dependencies
3. Initialize the vector store
4. Add .env file to the root of the project
5. Run the server
6. Run the user interface

To initilzie the vector store you need to run the following command ( you need to be in the root of the project)
```
python -m src.helpers.init_vectorstore
```

To add .env file to the root of the project you need to copy the .env.example file to .env

To run the server you need to run the following command ( you need to be in the root of the project)
```
python -m uvicorn src.server.app:app --reload
```

To run the user interface you need to run the following command ( you need to be in the root of the project)
```
cd src/user_interface
python ./app.py
```