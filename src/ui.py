# src/ui.py - Enhanced Gradio Interface
import gradio as gr
import os
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
import base64
from PIL import Image
import io
from pathlib import Path

def create_file_upload_interface(agent):
    """Create file upload and processing interface"""
    
    def process_uploaded_files(files, progress=gr.Progress()):
        """Process uploaded files and add to vector store"""
        if not files:
            return "❌ No files uploaded", "", ""
        
        progress(0, desc="Starting file processing...")
        
        results = []
        total_chunks = 0
        
        for i, file in enumerate(files):
            file_path = file.name
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            progress((i + 1) / len(files), desc=f"Processing {file_name}...")
            
            try:
                # Determine file type and process accordingly
                chunks_added = 0
                
                if file_ext == '.pdf':
                    chunks_added = agent.process_single_file(file_path, 'pdf')
                elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']:
                    chunks_added = agent.process_single_file(file_path, 'image')
                elif file_ext in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']:
                    chunks_added = agent.process_single_file(file_path, 'audio')
                else:
                    results.append(f"❌ {file_name}: Unsupported format")
                    continue
                
                if chunks_added > 0:
                    results.append(f"✅ {file_name}: {chunks_added} chunks created")
                    total_chunks += chunks_added
                else:
                    results.append(f"⚠️ {file_name}: No content extracted")
                    
            except Exception as e:
                results.append(f"❌ {file_name}: Error - {str(e)}")
        
        # Update stats
        stats = agent.get_stats()
        stats_text = f"""📊 **Current System Stats:**
- 📄 PDFs: {stats['pdf_count']}
- 🖼️ Images: {stats['img_count']} 
- 🎵 Audio: {stats['audio_count']}
- 📚 Document chunks: {stats['doc_chunks']}
- 🤖 Model: {stats['model']}"""
        
        processing_results = "\n".join(results)
        if total_chunks > 0:
            processing_results += f"\n\n🎉 **Total: {total_chunks} new chunks added to knowledge base**"
        
        return processing_results, stats_text, ""
    
    def ask_question(question, history, quick_mode=False):
        """Process question and return response"""
        if not question.strip():
            return history, "", None
        
        # Get response from agent
        response, sources, metadata, images = agent.ask_question(question, quick_mode)
        
        # Format response with sources
        formatted_response = response
        if sources:
            formatted_response += f"\n\n**📚 Sources:**\n"
            for i, source in enumerate(sources, 1):
                formatted_response += f"{i}. {os.path.basename(source)}\n"
        
        # Add to chat history
        history.append([question, formatted_response])
        
        # Prepare images for display
        image_gallery = []
        if images:
            for img in images:
                try:
                    # Try to load image from path first
                    if os.path.exists(img['path']):
                        image_gallery.append((img['path'], img['display_info']['title']))
                    # Fallback to base64 data if available
                    elif img.get('data'):
                        # Decode base64 and save temporarily
                        img_data = base64.b64decode(img['data'])
                        temp_path = f"temp_img_{len(image_gallery)}.png"
                        with open(temp_path, "wb") as f:
                            f.write(img_data)
                        image_gallery.append((temp_path, img['display_info']['title']))
                except Exception as e:
                    print(f"⚠️ Could not prepare image for display: {e}")
                    continue
        
        return history, "", image_gallery
    
    def clear_chat():
        """Clear chat history"""
        agent.qa_chain.clear_history() if agent.qa_chain else None
        return [], "", None
    
    def switch_model(model_name):
        """Switch AI model"""
        success = agent.switch_model(model_name)
        if success:
            return f"✅ Successfully switched to {model_name}"
        else:
            return f"❌ Failed to switch to {model_name}. Make sure it's installed with: ollama pull {model_name}"
    
    # Create Gradio interface
    with gr.Blocks(
        title="🤖 Offline AI Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px !important;
        }
        #image-gallery {
            border: 2px dashed #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        #image-gallery img {
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 Offline AI Agent
        
        **Upload documents, images, or audio files and ask questions about their content!**
        
        ⚡ **Optimized for speed** - Uses local AI models for complete privacy
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # File upload section
                gr.Markdown("## 📁 Upload Files")
                
                file_upload = gr.File(
                    label="Upload Documents, Images, or Audio Files",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", 
                               ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"]
                )
                
                process_btn = gr.Button("🔄 Process Files", variant="primary", size="lg")
                
                processing_output = gr.Textbox(
                    label="Processing Results",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                # Stats and controls
                gr.Markdown("## 📊 System Status")
                
                stats_display = gr.Textbox(
                    label="Current Stats",
                    lines=8,
                    interactive=False,
                    value=f"""📊 **System Stats:**
- 📄 PDFs: 0
- 🖼️ Images: 0
- 🎵 Audio: 0
- 📚 Document chunks: 0
- 🤖 Model: {agent.model_name}"""
                )
                
                # Model switching
                gr.Markdown("### 🔄 Switch Model")
                model_dropdown = gr.Dropdown(
                    choices=["phi3:mini", "mistral", "llama3", "codellama"],
                    value=agent.model_name,
                    label="AI Model"
                )
                switch_btn = gr.Button("Switch Model", size="sm")
                model_status = gr.Textbox(label="Model Status", lines=2, interactive=False)
        
        # Chat interface
        gr.Markdown("## 💬 Chat with Your Documents")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="AI Assistant",
                    height=400,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                # Image gallery for relevant images
                image_gallery = gr.Gallery(
                    label="📸 Relevant Images",
                    show_label=True,
                    elem_id="image-gallery",
                    columns=3,
                    rows=2,
                    height="300px",
                    object_fit="contain",
                    visible=False
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="What would you like to know about your documents?",
                        lines=2,
                        scale=4
                    )
                    
                with gr.Row():
                    ask_btn = gr.Button("💬 Ask", variant="primary", scale=2)
                    quick_btn = gr.Button("⚡ Quick Ask", variant="secondary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear", variant="stop", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Tips")
                gr.Markdown("""
                **Quick Start:**
                1. Upload your files above
                2. Click "Process Files"
                3. Ask questions below!
                
                **Supported Files:**
                - 📄 PDFs
                - 🖼️ Images (with OCR)
                - 🎵 Audio (with transcription)
                
                **Models:**
                - **phi3:mini**: Fastest responses
                - **mistral**: Balanced quality/speed
                - **llama3**: Highest quality
                
                **Quick Ask**: Ultra-fast responses with minimal context
                
                
                """)
        
        # Function to show/hide image gallery based on content
        def update_image_gallery_visibility(images):
            if images and len(images) > 0:
                return gr.update(visible=True, value=images)
            else:
                return gr.update(visible=False, value=[])
        
        # Event handlers
        process_btn.click(
            fn=process_uploaded_files,
            inputs=[file_upload],
            outputs=[processing_output, stats_display, question_input],
            show_progress=True
        )
        
        ask_btn.click(
            fn=lambda q, h: ask_question(q, h, quick_mode=False),
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, image_gallery]
        ).then(
            fn=update_image_gallery_visibility,
            inputs=[image_gallery],
            outputs=[image_gallery]
        )
        
        quick_btn.click(
            fn=lambda q, h: ask_question(q, h, quick_mode=True),
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, image_gallery]
        ).then(
            fn=update_image_gallery_visibility,
            inputs=[image_gallery],
            outputs=[image_gallery]
        )
        
        question_input.submit(
            fn=lambda q, h: ask_question(q, h, quick_mode=False),
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, image_gallery]
        ).then(
            fn=update_image_gallery_visibility,
            inputs=[image_gallery],
            outputs=[image_gallery]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, question_input, image_gallery]
        )
        
        switch_btn.click(
            fn=switch_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        # Load initial stats
        def load_initial_stats():
            stats = agent.get_stats()
            return f"""📊 **System Stats:** (Fresh Start)
- 📄 PDFs: {stats['pdf_count']}
- 🖼️ Images: {stats['img_count']}
- 🎵 Audio: {stats['audio_count']}
- 📚 Document chunks: {stats['doc_chunks']}
- 🤖 Model: {stats['model']}"""
        
        demo.load(fn=load_initial_stats, outputs=[stats_display])
    
    return demo

def launch_gradio_interface(agent):
    """Launch the Gradio web interface"""
    return create_file_upload_interface(agent)

def launch_terminal_interface(agent):
    """Launch terminal-based chat interface"""
    print("\n🚀 Terminal Chat Interface")
    print("=" * 50)
    print("💡 Commands:")
    print("   'quit' or 'exit' - Exit the chat")
    print("   'clear' - Clear conversation history")
    print("   'stats' - Show system statistics")
    print("   'switch phi3' - Switch to phi3:mini model")
    print("   'switch mistral' - Switch to mistral model")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif question.lower() == 'clear':
                agent.qa_chain.clear_history() if agent.qa_chain else None
                print("🗑️ Chat history cleared")
                continue
            
            elif question.lower() == 'stats':
                stats = agent.get_stats()
                print(f"\n📊 System Statistics:")
                print(f"🤖 Model: {stats['model']}")
                print(f"📄 PDFs: {stats['pdf_count']}")
                print(f"🖼️ Images: {stats['img_count']}")
                print(f"🎵 Audio files: {stats['audio_count']}")
                print(f"📚 Document chunks: {stats['doc_chunks']}")
                print(f"📁 Unique sources: {stats['sources']}")
                continue
            
            elif question.lower().startswith('switch '):
                model_name = question.lower().replace('switch ', '').strip()
                if model_name in ['phi3', 'phi3:mini']:
                    model_name = 'phi3:mini'
                success = agent.switch_model(model_name)
                if success:
                    print(f"✅ Switched to {model_name}")
                else:
                    print(f"❌ Failed to switch to {model_name}")
                continue
            
            # Process question
            print("🤔 Thinking...")
            response, sources, metadata, images = agent.ask_question(question)
            
            print(f"\n🤖 **Answer:**")
            print(response)
            
            if sources:
                print(f"\n📚 **Sources:**")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {os.path.basename(source)}")
            
            if images:
                print(f"\n🖼️ **Relevant Images:** {len(images)} found")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

# Test the UI components
if __name__ == "__main__":
    print("🧪 Testing UI Components...")
    print("💡 This module provides both Gradio web interface and terminal chat interface")
    print("   Use main.py to launch the full application")