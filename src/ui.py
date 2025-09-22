# src/ui.py
import gradio as gr
from typing import Any, List, Tuple
import os
import shutil
import tempfile

def launch_gradio_interface(agent: Any):
    """Launch Gradio web interface"""
    
    # Store uploaded files temporarily
    uploaded_files = []
    
    def chat_interface(message, history, show_images):
        """Chat interface function"""
        if not message.strip():
            return history, ""
        
        try:
            response, sources, metadata, relevant_images = agent.ask_question(message)
            
            # Add images to response if requested and available
            if show_images and relevant_images:
                response += f"\n\n🖼️ **Relevant Images ({len(relevant_images)} found):**"
                for img in relevant_images[:3]:  # Limit to 3 images
                    response += f"\n• {img['filename']} (from {img['source']}, Page {img['page']})"
            
            history.append([message, response])
            
            # Return images for display if requested
            image_paths = []
            if show_images and relevant_images:
                image_paths = [img['path'] for img in relevant_images[:3]]
            
            return history, "", image_paths
            
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            history.append([message, error_response])
            return history, "", []
    
    def upload_files(files):
        """Handle file uploads"""
        nonlocal uploaded_files
        if not files:
            return "No files selected"
        
        uploaded_count = 0
        messages = []
        
        for file in files:
            try:
                # Handle both file path string and file object
                if hasattr(file, 'name'):
                    file_path = file.name
                    filename = os.path.basename(file_path)
                else:
                    file_path = file
                    filename = os.path.basename(file_path)
                
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
                
                # Copy file using shutil for better reliability
                shutil.copy2(file_path, dest_path)
                
                # Track uploaded files
                uploaded_files.append({
                    'path': str(dest_path),
                    'type': file_ext,
                    'filename': filename
                })
                
                uploaded_count += 1
                messages.append(f"✅ Uploaded: {filename}")
                
            except Exception as e:
                messages.append(f"❌ Error uploading {getattr(file, 'name', str(file))}: {str(e)}")
        
        result = f"📁 Uploaded {uploaded_count} files\n" + "\n".join(messages)
        if uploaded_count > 0:
            result += f"\n\n💡 Click 'Process Documents' to analyze these files"
        return result
    
    def process_uploaded_documents():
        """Process only the uploaded documents"""
        nonlocal uploaded_files
        try:
            if not uploaded_files:
                return "❌ No files uploaded yet. Please upload files first."
            
            # Initialize components if needed
            agent._initialize_components()
            
            # Process uploaded files by type
            all_chunks = []
            processed_count = 0
            
            # Group files by type
            pdf_files = [f for f in uploaded_files if f['type'] == '.pdf']
            image_files = [f for f in uploaded_files if f['type'] in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']]
            audio_files = [f for f in uploaded_files if f['type'] in ['.wav', '.mp3', '.m4a', '.flac', '.aac']]
            
            # Process PDFs
            if pdf_files:
                for pdf_file in pdf_files:
                    chunks = agent.doc_processor.process_pdf(pdf_file['path'])
                    if chunks:
                        agent.vector_store.add_documents(chunks, "pdf")
                        all_chunks.extend(chunks)
                        processed_count += 1
            
            # Process Images
            if image_files:
                for img_file in image_files:
                    chunks = agent.image_processor.process_image(img_file['path'])
                    if chunks:
                        agent.vector_store.add_documents(chunks, "image")
                        all_chunks.extend(chunks)
                        processed_count += 1
            
            # Process Audio
            if audio_files:
                for audio_file in audio_files:
                    chunks = agent.audio_processor.process_audio(audio_file['path'])
                    if chunks:
                        agent.vector_store.add_documents(chunks, "audio")
                        all_chunks.extend(chunks)
                        processed_count += 1
            
            # Save the vector store
            if all_chunks:
                agent.vector_store.save()
                result = f"✅ Successfully processed {processed_count} files!\n"
                result += f"📊 Created {len(all_chunks)} text chunks\n"
                result += f"🔍 Ready to answer questions about your content"
                
                # Clear uploaded files list since they're now processed
                uploaded_files = []
                return result
            else:
                return "⚠️ No content could be extracted from the uploaded files"
                
        except Exception as e:
            return f"❌ Error processing documents: {str(e)}"
    
    def process_documents():
        """Process documents function - prioritize uploaded files"""
        try:
            # Always try uploaded files first
            if uploaded_files:
                return process_uploaded_documents()
            
            # Only check data directories if no files uploaded
            has_files = False
            pdf_dir = agent.data_dir / "pdfs"
            img_dir = agent.data_dir / "images"
            audio_dir = agent.data_dir / "audio"
            
            if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
                has_files = True
            if img_dir.exists() and any(f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'] for f in img_dir.glob("*")):
                has_files = True
            if audio_dir.exists() and any(f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'] for f in audio_dir.glob("*")):
                has_files = True
            
            if has_files:
                agent.process_documents()
                return "✅ Documents from data folders processed successfully! You can now ask questions about your content."
            else:
                return "⚠️ No files found to process. Please upload files using the file uploader above, then click 'Process Documents'."
                
        except Exception as e:
            return f"❌ Error processing documents: {str(e)}"
    
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
                
                # Image display area
                image_gallery = gr.Gallery(
                    label="🖼️ Relevant Images",
                    show_label=True,
                    elem_id="image-gallery",
                    columns=3,
                    rows=1,
                    height="auto",
                    visible=False
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
                
                with gr.Row():
                    show_images_checkbox = gr.Checkbox(
                        label="Show relevant images in responses",
                        value=False
                    )
            
            with gr.Column(scale=1):
                # Control panel
                gr.Markdown("## 🛠️ Control Panel")
                
                # File upload
                file_upload = gr.File(
                    label="📁 Upload Files",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".wav", ".mp3", ".m4a", ".flac"],
                    file_count="multiple",
                    type="filepath"
                )
                upload_btn = gr.Button("Upload Files 📤", variant="secondary")
                upload_status = gr.Textbox(label="Upload Status", lines=3, interactive=False)
                
                # Process documents
                process_btn = gr.Button("Process Documents 🔄", variant="primary")
                process_status = gr.Textbox(label="Processing Status", lines=2, interactive=False)
                
                # Clear uploaded files
                clear_files_btn = gr.Button("Clear Uploaded Files 🗑️", variant="secondary")
                
                # System status
                status_btn = gr.Button("System Status 📊", variant="secondary")
                system_status = gr.Textbox(label="System Status", lines=8, interactive=False)
        
        # Event handlers
        def submit_message(message, history, show_images):
            new_history, empty_msg, images = chat_interface(message, history, show_images)
            # Show/hide image gallery based on whether images are found
            gallery_visible = len(images) > 0 if images else False
            return new_history, empty_msg, images, gr.update(visible=gallery_visible)
        
        def clear_chat():
            return [], gr.update(visible=False)
        
        def clear_uploaded_files():
            nonlocal uploaded_files
            uploaded_files = []
            return "🗑️ Cleared all uploaded files. Upload new files to process."
        
        # Chat events
        msg.submit(
            submit_message, 
            inputs=[msg, chatbot, show_images_checkbox], 
            outputs=[chatbot, msg, image_gallery, image_gallery]
        )
        send_btn.click(
            submit_message, 
            inputs=[msg, chatbot, show_images_checkbox], 
            outputs=[chatbot, msg, image_gallery, image_gallery]
        )
        clear_btn.click(clear_chat, outputs=[chatbot, image_gallery])
        
        # File upload events
        upload_btn.click(upload_files, inputs=[file_upload], outputs=[upload_status])
        
        # Processing events
        process_btn.click(process_documents, outputs=[process_status])
        
        # Clear files events
        clear_files_btn.click(clear_uploaded_files, outputs=[upload_status])
        
        # Status events
        status_btn.click(get_system_status, outputs=[system_status])
        
        # Welcome message
        gr.Markdown("""
        ## 🚀 Getting Started
        
        1. **Upload your files** using the file uploader above (PDFs, images, audio)
        2. **Click "Process Documents"** to analyze your content
        3. **Ask questions** about your uploaded content
        4. **Check "System Status"** to see what's been processed
        5. **Enable "Show relevant images"** to see document images in responses
        
        ### 💡 How It Works
        - **Upload files first** using the file uploader
        - **Then click "Process Documents"** to analyze them
        - Files are processed **immediately** - no restart needed
        - Upload multiple files at once for batch processing
        - Use "Clear Uploaded Files" to reset and upload new files
        
        ### ⚠️ Important Notes
        - **Always upload files first** before clicking "Process Documents"
        - The system processes **uploaded files**, not files in data/ folders
        - If no files are uploaded, you'll get a "no files found" message
        
        ### 📝 Supported Formats
        - **Documents:** PDF
        - **Images:** PNG, JPG, JPEG, BMP, TIFF (with OCR)
        - **Audio:** WAV, MP3, M4A, FLAC (with speech-to-text)
        
        ### 🖼️ Image Features
        - **PDF Images:** Automatically extracted and displayed when relevant
        - **Strong Citations:** Detailed source references with page numbers
        - **Visual Context:** See the actual images from your documents
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