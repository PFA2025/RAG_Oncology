import gradio as gr
import requests

# Configuration
API_URL = "http://127.0.0.1:8000"

def chat_with_agent(message, history):
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

def create_memory(name, description):
    """Create a new user memory"""
    try:
        response = requests.post(
            f"{API_URL}/user-memories/",
            json={"name": name, "description": description}
        )
        response.raise_for_status()
        return "Memory created successfully!"
    except Exception as e:
        return f"Error creating memory: {str(e)}"

def get_memory(memory_id):
    """Retrieve a specific user memory"""
    try:
        response = requests.get(f"{API_URL}/user-memories/{memory_id}")
        response.raise_for_status()
        memory = response.json()
        return f"Name: {memory.get('name')}\nDescription: {memory.get('description')}"
    except Exception as e:
        return f"Error retrieving memory: {str(e)}"

def create_chat_interface():
    """Create the chat interface"""
    with gr.Blocks(title="Cancer Agent Interface") as demo:
        gr.Markdown("# Cancer Agent Interface")
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your Message")
            clear = gr.Button("Clear")
            
            def respond(message, chat_history):
                bot_message = chat_with_agent(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history
            
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Tab("Memory Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Create New Patient")
                    patient_name = gr.Textbox(label="Patient Name")
                    patient_desc = gr.Textbox(label="Patient Description", lines=3)
                    create_btn = gr.Button("Create Patient")
                    create_output = gr.Textbox(label="Status", interactive=False)
                    
                    create_btn.click(
                        create_memory,
                        inputs=[patient_name, patient_desc],
                        outputs=create_output
                    )
                
                with gr.Column():
                    gr.Markdown("### View Patient")
                    patient_id = gr.Number(label="Patient ID", value=1)
                    view_btn = gr.Button("View Patient")
                    patient_output = gr.Textbox(label="Patient Details", lines=5, interactive=False)
                    
                    view_btn.click(
                        get_memory,
                        inputs=patient_id,
                        outputs=patient_output
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
