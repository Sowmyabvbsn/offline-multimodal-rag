# src/ui.py
import gradio as gr
from typing import Any, List, Tuple
import os

def launch_gradio_interface(agent: Any):
    """Launch Gradio web interface"""
    
    def chat_interface(message, history):
        """Chat interface function"""
        if not message.strip():
            return history, ""
        
        try:
            response, sources = agent.ask_question(message)
            
            # Format response with sources
            if sources:
                unique_sources = list(dict.fromkeys(sources))  # Remove duplicates
                source_names = [os.path.basename(src) for src in unique_sources]
                response += f"\n\n📚 **Sources:** {', '.join(source_names)}"
            
            history.append([message, response])
            return history, ""
            
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            history.append([message, error_response])
            return history, ""
    
    def process_documents():
        """Process documents function"""
        try:
            agent.process_documents()
            return "✅ Documents processed successfully! You can now ask questions about your content."
        except Exception as e:
            return f"❌ Error processing documents: {str(e)}"
    
    def upload_file(files):
        """Handle file uploads"""
        if not files:
            return "No files selected"
        
        uploaded_count = 0
        messages = []
        
        for file in files:
            try:
                filename = os.path.basename(file.name)
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Determine destination folder
                if file_ext == '.pdf':
                    dest_folder = agent.data_dir / "pdfs"
                elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
                    dest_folder = agent.data_dir / "images"
                elif file_ext in ['.wav', '.mp3', '.m4a', '.flac', '.aac']:
                    dest_folder = agent.data_dir / "audio"
                else:
                    messages.append(f"⚠️ Unsupported file type: {filename}")
                    continue
                
                # Copy file to destination
                dest_path = dest_folder / filename
                dest_folder.mkdir(parents=True, exist_ok=True)
                
                # Read and write file
                with open(file.name, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                
                uploaded_count += 1
                messages.append(f"✅ Uploaded: {filename}")
                
            except Exception as e:
                messages.append(f"❌ Error uploading {filename}: {str(e)}")
        
        result = f"📁 Uploaded {uploaded_count} files\n" + "\n".join(messages)
        return result
    
    def get_system_status():
        """Get current system status"""
        try:
            # Get document counts
            pdf_count = len(list((agent.data_dir / "pdfs").glob("*.pdf"))) if (agent.data_dir / "pdfs").exists() else 0
            img_count = len(list((agent.data_dir / "images").glob("*"))) if (agent.data_dir / "images").exists() else 0
            audio_count = len(list((agent.data_dir / "audio").glob("*"))) if (agent.data_dir / "audio").exists() else 0
            
            # Get vector store stats
            if hasattr(agent, 'vector_store') and agent.vector_store:
                stats = agent.vector_store.get_stats()
                doc_chunks = stats.get('total_documents', 0)
                sources = len(stats.get('sources', []))
            else:
                doc_chunks = 0
                sources = 0
            
            status = f"""📊 **System Status**
            
📁 **Files Available:**
• PDFs: {pdf_count}
• Images: {img_count}  
• Audio: {audio_count}

🔍 **Processed Content:**
• Document chunks: {doc_chunks}
• Unique sources: {sources}

💡 **Ready to answer questions!**"""
            
            return status
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"
    
    def clear_chat():
        """Clear chat history"""
        return []
    
    # Create Gradio interface
    with gr.Blocks(
        title="Offline AI Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px !important;
        }
        """
    ) as demo:
        
        gr.Markdown("# 🤖 Offline AI Agent")
        gr.Markdown("Ask questions about your documents, images, and audio files - completely offline!")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="💬 Chat with your documents",
                    height=500,
                    show_label=True,
                    container=True,
                    elem_classes=["chat-container"]
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask me anything about your uploaded documents...",
                        scale=4,
                        lines=1
                    )
                    send_btn = gr.Button("Send 📤", scale=1, variant="primary")
                    clear_btn = gr.Button("Clear 🗑️", scale=1)
            
            with gr.Column(scale=1):
                # Control panel
                gr.Markdown("## 🛠️ Control Panel")
                
                # File upload
                file_upload = gr.Files(
                    label="📁 Upload Files",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".wav", ".mp3", ".m4a", ".flac"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("Upload Files 📤", variant="secondary")
                upload_status = gr.Textbox(label="Upload Status", lines=3, interactive=False)
                
                # Process documents
                process_btn = gr.Button("Process Documents 🔄", variant="primary")
                process_status = gr.Textbox(label="Processing Status", lines=2, interactive=False)
                
                # System status
                status_btn = gr.Button("System Status 📊", variant="secondary")
                system_status = gr.Textbox(label="System Status", lines=8, interactive=False)
        
        # Event handlers
        def submit_message(message, history):
            return chat_interface(message, history)
        
        # Chat events
        msg.submit(submit_message, inputs=[msg, chatbot], outputs=[chatbot, msg])
        send_btn.click(submit_message, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot])
        
        # File upload events
        upload_btn.click(upload_file, inputs=[file_upload], outputs=[upload_status])
        
        # Processing events
        process_btn.click(process_documents, outputs=[process_status])
        
        # Status events
        status_btn.click(get_system_status, outputs=[system_status])
        
        # Welcome message
        gr.Markdown("""
        ## 🚀 Getting Started
        
        1. **Upload your files** using the file uploader (PDFs, images, audio)
        2. **Click "Process Documents"** to analyze your content
        3. **Ask questions** about your uploaded content
        4. **Check "System Status"** to see what's been processed
        
        ### 📝 Supported Formats
        - **Documents:** PDF
        - **Images:** PNG, JPG, JPEG, BMP, TIFF (with OCR)
        - **Audio:** WAV, MP3, M4A, FLAC (with speech-to-text)
        """)
    
    return demo

def launch_terminal_interface(agent: Any):
    """Launch terminal chat interface"""
    print("\n" + "="*60)
    print("🤖 Offline AI Agent - Terminal Interface")
    print("="*60)
    print("Commands:")
    print("  'help' - Show this help")
    print("  'status' - Show system status") 
    print("  'process' - Process documents")
    print("  'clear' - Clear conversation history")
    print("  'quit' or 'exit' - Exit the program")
    print("="*60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif question.lower() == 'help':
                print("""
📋 Available Commands:
• help - Show this help message
• status - Show system status and file counts
• process - Process all documents in data folders
• clear - Clear conversation history
• quit/exit - Exit the program

💡 Just type any question to chat with your documents!
                """)
                
            elif question.lower() == 'status':
                try:
                    # Get file counts
                    pdf_count = len(list((agent.data_dir / "pdfs").glob("*.pdf"))) if (agent.data_dir / "pdfs").exists() else 0
                    img_count = len(list((agent.data_dir / "images").glob("*"))) if (agent.data_dir / "images").exists() else 0
                    audio_count = len(list((agent.data_dir / "audio").glob("*"))) if (agent.data_dir / "audio").exists() else 0
                    
                    print(f"""
📊 System Status:
📁 Files: {pdf_count} PDFs, {img_count} images, {audio_count} audio files
🔍 Vector store: {agent.vector_store.get_stats()['total_documents'] if hasattr(agent, 'vector_store') else 0} chunks
                    """)
                except Exception as e:
                    print(f"❌ Error getting status: {e}")
                    
            elif question.lower() == 'process':
                try:
                    print("🔄 Processing documents...")
                    agent.process_documents()
                    print("✅ Processing complete!")
                except Exception as e:
                    print(f"❌ Error processing: {e}")
                    
            elif question.lower() == 'clear':
                if hasattr(agent, 'qa_chain') and agent.qa_chain:
                    agent.qa_chain.clear_history()
                print("🗑️ Conversation history cleared")
                
            else:
                # Regular question
                try:
                    response, sources = agent.ask_question(question)
                    print(f"\n🤖 Answer: {response}")
                    
                    if sources:
                        unique_sources = list(dict.fromkeys(sources))
                        source_names = [os.path.basename(src) for src in unique_sources]
                        print(f"📚 Sources: {', '.join(source_names)}")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

# Test the UI components
if __name__ == "__main__":
    print("🧪 Testing UI components...")
    print("✅ Gradio interface functions defined")
    print("✅ Terminal interface functions defined")
    print("💡 Import this module and call launch_gradio_interface(agent) or launch_terminal_interface(agent)")