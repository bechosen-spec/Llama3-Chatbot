# 🦙💬 Llama 3 Chatbot with Streamlit

Welcome to the Llama 3 Chatbot application! This project leverages the power of Hugging Face's Llama 3 model to create an interactive chatbot interface using Streamlit.

## Features

- **Interactive Chat:** Engage in a conversation with the Llama 3 model.
- **Customizable System Prompts:** Modify the system prompt to guide the chatbot's responses.
- **Session Management:** Maintains chat history for the session.
- **Clear Chat History:** Easily reset the chat history with a single click.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Llama3-Chatbot.git
    cd Llama3-Chatbot
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your Hugging Face API token:**

   Replace `"your-huggingface-api-token"` in the `app.py` file with your actual Hugging Face API token. You can obtain the token from your Hugging Face account settings.

### Running the Application

To start the Streamlit app, navigate to the project directory and run:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

## Usage

1. **Open the Streamlit app in your web browser.**
2. **Enter a system prompt** in the sidebar to guide the chatbot's behavior.
3. **Type your messages** in the chat input box and press Enter to send.
4. **View the chatbot's responses** as they appear in the chat window.
5. **Clear the chat history** by clicking the "Clear Chat History" button in the sidebar if needed.

## Project Structure

- `app.py`: The main application script that sets up the Streamlit interface and handles chat interactions.
- `requirements.txt`: Lists the required Python packages for the project.

## Troubleshooting

- **Model Access Issues:** Ensure your Hugging Face API token is correctly set in the `app.py` file. Check your internet connection if you face issues accessing the model.
- **Dependency Issues:** Run `pip install -r requirements.txt` to ensure all dependencies are installed.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Hugging Face for providing the Llama 3 model.
- Streamlit for the easy-to-use web application framework.

## Contact

For any questions or suggestions, please reach out to [Boniface Emmanuel](https://www.linkedin.com/in/emmanuel-boniface/).

Thank you for using the Llama 3 Chatbot! Happy chatting! 🦙💬

