import gradio as gr
import requests

# Configuration
API_URL = "http://127.0.0.1:8000"

def chat_with_agent(message, history, user_id):
    """Send message to the chat endpoint and return the response"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message}
        )
        response.raise_for_status()
        return response.json().get("response", "No response from server")
    except Exception as e:
        return f"Error: {str(e)}"

def create_memory(user_id, name, description):
    """Create a new user memory"""
    try:
        response = requests.post(
            f"{API_URL}/user-memories/",
            json={"user_id": user_id, "name": name, "description": description}
        )
        response.raise_for_status()
        return "Patient profile created successfully!"
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg = e.response.json().get('detail', error_msg)
        return f"Error creating patient profile: {error_msg}"

def get_memory(user_id):
    """Retrieve a specific user memory by user ID"""
    try:
        response = requests.get(f"{API_URL}/user-memories/user/{user_id}")
        response.raise_for_status()
        memory = response.json()
        return (
            f"Patient ID: {memory.get('user_id')}\n"
            f"Name: {memory.get('name', 'N/A')}\n"
            f"Description: {memory.get('description', 'N/A')}\n"
            f"Last Updated: {memory.get('updated_at', 'N/A')}"
        )
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg = e.response.json().get('detail', error_msg)
        return f"Error retrieving patient profile: {error_msg}"

def create_chat_interface():
    """Create the chat interface"""
    with gr.Blocks(title="Cancer Agent Interface") as demo:
        # Store the current patient ID in a state variable
        current_patient_id = gr.State(value=1)  # Default to patient ID 1
        
        gr.Markdown("# Cancer Agent Interface")
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(type="messages")  # Updated to use new message format
            msg = gr.Textbox(label="Your Message")
            clear = gr.Button("Clear")
            
            def respond(message, chat_history, user_id):
                if not message.strip():
                    return "", chat_history
                
                # Add user message to chat history
                chat_history.append({"role": "user", "content": message})
                
                # Get bot response
                bot_message = chat_with_agent(message, chat_history, user_id)
                
                # Add bot response to chat history
                chat_history.append({"role": "assistant", "content": bot_message})
                
                return "", chat_history
            
            msg.submit(
                respond,
                inputs=[msg, chatbot, current_patient_id],
                outputs=[msg, chatbot]
            )
            clear.click(lambda: [], None, chatbot, queue=False)
        
        with gr.Tab("Memory Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Create/Update Patient Profile")
                    patient_id = gr.Number(label="Patient ID", value=1, precision=0)
                    patient_name = gr.Textbox(label="Patient Name")
                    patient_desc = gr.TextArea(label="Patient Description", lines=3)
                    
                    with gr.Row():
                        create_btn = gr.Button("Save Profile")
                        clear_btn = gr.Button("Clear Form")
                        
                    create_output = gr.Textbox(label="Status", interactive=False)
                    
                    def save_profile(user_id, name, description):
                        return create_memory(user_id, name, description)
                        
                    def clear_form():
                        return [None, "", "", ""]
                    
                    create_btn.click(
                        save_profile,
                        inputs=[patient_id, patient_name, patient_desc],
                        outputs=create_output
                    )
                    
                    clear_btn.click(
                        clear_form,
                        outputs=[patient_id, patient_name, patient_desc, create_output]
                    )
                
                with gr.Column():
                    gr.Markdown("### View Patient Profile")
                    view_id = gr.Number(label="Patient ID", value=1, precision=0)
                    view_btn = gr.Button("Load Profile")
                    patient_output = gr.Textbox(label="Patient Details", lines=8, interactive=False)
                    
                    def load_profile(user_id):
                        result = get_memory(user_id)
                        if result.startswith("Error"):
                            return [user_id, "", "", result]  # Keep the user_id even on error
                        # Parse the response to pre-fill the form
                        try:
                            lines = result.split('\n')
                            user_id = int(lines[0].split(': ')[1])
                            name = lines[1].split(': ')[1] if len(lines) > 1 and ': ' in lines[1] else ""
                            desc = lines[2].split(': ')[1] if len(lines) > 2 and ': ' in lines[2] else ""
                            return [user_id, name, desc, result] + [user_id]  # Also update the state
                        except Exception as e:
                            return [user_id, "", "", f"Error parsing profile: {str(e)}", user_id]
                    
                    view_btn.click(
                        load_profile,
                        inputs=view_id,
                        outputs=[patient_id, patient_name, patient_desc, patient_output]
                    )
        
        with gr.Tab("API Status"):
            status_btn = gr.Button("Check API Status")
            status_output = gr.Textbox(label="API Status", interactive=False)
            
            def check_status():
                try:
                    response = requests.get(f"{API_URL}/health")
                    if response.status_code == 200:
                        return "✅ API is running and healthy!"
                    return f"⚠️ API returned status code: {response.status_code}"
                except Exception as e:
                    return f"❌ Could not connect to API: {str(e)}"
            
            status_btn.click(check_status, outputs=status_output)
    
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
