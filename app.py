import streamlit as st
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

st.set_page_config(page_title="🦙💬 Llama 2 Chatbot with Streamlit")

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with the actual model ID
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0})

    return tokenizer, model

tokenizer, model = get_tokenizer_model()

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

    def generate_llama2_response(user_prompt, system_prompt):
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

    # Generate Llama 2 response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt, system_prompt)
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
            {"role": "assistant", "content": "How may I assist you today"}]

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if __name__ == '__main__':
    main()
