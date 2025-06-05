# Final_year Project_2025
This is repo for PFA 2025

## How to run the project
1. Clone the repository
2. Install the dependencies
3. Run the server
4. Run the user interface

You need first to run the server via this command ( you need to be in the root of the project)
```
python -m uvicorn src.server.app:app --reload
```

after that you need to run the user interface via this command
```
cd src/user_interface
python ./app.py
```