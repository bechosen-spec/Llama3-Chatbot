import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

st.set_page_config(page_title="🦙💬 Llama 3 Chatbot with Streamlit")

# Load model directly without caching for debugging purposes
def get_tokenizer_model():
    try:
        # Create tokenizer
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with the actual model ID
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = get_tokenizer_model()

if tokenizer is None or model is None:
    st.stop()

def main():
    st.title('Boniface Emmanuel ChatBot')

    # Create a Sidebar
    with st.sidebar:
        st.title("At Your Service")

        # User inputs for system prompts
        system_prompt = st.text_input("Enter System Prompt")

    # Initialize chat messages session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}]

    def generate_llama3_response(user_prompt, system_prompt):
        runtime_flag = "cuda:0"

        # Create TextStreamer for efficient memory handling during generation
        text_streamer = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Prepare input for the model
        input_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], add_generation_prompt=True, return_tensors="pt"
        ).to(runtime_flag)

        # Define termination conditions
        terminators = [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("")]

        # Generate text using the model and streamer
        outputs = model.generate(
            input_ids,
            max_new_tokens=2056,  # Adjust as needed
            eos_token_id=terminators,
            do_sample=True,
            streamer=text_streamer
        )

        # Decode the generated tokens into text
        output_text = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        return output_text

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for a chat prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate Llama 3 response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama3_response(prompt, system_prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        # Store the assistant's response in the chat messages
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

    # Button to clear chat history
    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}]

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if __name__ == '__main__':
    main()
