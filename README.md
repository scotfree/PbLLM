# PbLLM


Here are the steps to start the app:
First, install all required dependencies:
txt
Create a .env file in your project root with your Google OAuth credentials and OpenAI API key:
your_openai_api_key_here
Make sure you have the templates directory with index.html in your project root.
4. Start the FastAPI server using uvicorn:
reload
The app will be available at http://localhost:8000. The --reload flag enables auto-reload when you make code changes.
Note: Before the app will work properly, you'll need to:
Set up Google OAuth credentials in the Google Cloud Console
Have a valid OpenAI API key
Create at least one user in the database (the app currently doesn't have a registration endpoint for non-Google users)
