# src/audio_processor.py
import whisper
from typing import List, Dict
import os
import math

class AudioProcessor:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.chunk_duration = 600  # 10 minutes in seconds
        print(f"🎵 Audio processor initialized (model: {model_size})")
    
    def _load_whisper_model(self):
        """Lazy load the Whisper model"""
        if self.model is None:
            print(f"📥 Loading Whisper model: {self.model_size}")
            try:
                self.model = whisper.load_model(self.model_size)
                print("✅ Whisper model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading Whisper model: {e}")
                raise
    
    def transcribe_audio(self, file_path: str) -> Dict:
        """Transcribe audio file to text with timestamps"""
        self._load_whisper_model()
        
        try:
            print(f"🎤 Transcribing audio: {os.path.basename(file_path)}")
            result = self.model.transcribe(file_path, verbose=False)
            
            transcript = result['text'].strip()
            print(f"✅ Transcription complete: {len(transcript)} characters")
            
            return {
                'text': transcript,
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', [])
            }
            
        except Exception as e:
            print(f"❌ Error transcribing {file_path}: {e}")
            return {'text': '', 'language': 'unknown', 'segments': []}
    
    def chunk_transcript(self, transcript_data: Dict, source_info: str, max_chunk_size: int = 1000) -> List[Dict]:
        """Split transcript into chunks with timestamps"""
        transcript = transcript_data['text']
        segments = transcript_data.get('segments', [])
        language = transcript_data.get('language', 'unknown')
        
        if not transcript.strip():
            return []
        
        chunks = []
        
        if segments:
            # Use segment-based chunking (more accurate with timestamps)
            chunks = self._chunk_by_segments(segments, source_info, language, max_chunk_size)
        else:
            # Fallback to sentence-based chunking
            chunks = self._chunk_by_sentences(transcript, source_info, language, max_chunk_size)
        
        print(f"📝 Created {len(chunks)} audio chunks from {os.path.basename(source_info)}")
        return chunks
    
    def _chunk_by_segments(self, segments: List[Dict], source_info: str, language: str, max_chunk_size: int) -> List[Dict]:
        """Create chunks based on Whisper segments with timestamps"""
        chunks = []
        current_chunk = ""
        current_start_time = 0
        current_end_time = 0
        
        for segment in segments:
            segment_text = segment.get('text', '').strip()
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            
            # If adding this segment would exceed max size, create a chunk
            if len(current_chunk + segment_text) > max_chunk_size and current_chunk:
                chunk = {
                    'text': current_chunk.strip(),
                    'source': source_info,
                    'chunk_id': len(chunks),
                    'type': 'audio',
                    'language': language,
                    'start_time': current_start_time,
                    'end_time': current_end_time,
                    'duration': current_end_time - current_start_time
                }
                chunks.append(chunk)
                current_chunk = segment_text
                current_start_time = segment_start
            else:
                if not current_chunk:  # First segment
                    current_start_time = segment_start
                current_chunk += " " + segment_text
            
            current_end_time = segment_end
        
        # Add the last chunk
        if current_chunk.strip():
            chunk = {
                'text': current_chunk.strip(),
                'source': source_info,
                'chunk_id': len(chunks),
                'type': 'audio',
                'language': language,
                'start_time': current_start_time,
                'end_time': current_end_time,
                'duration': current_end_time - current_start_time
            }
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, transcript: str, source_info: str, language: str, max_chunk_size: int) -> List[Dict]:
        """Fallback chunking method using sentences"""
        sentences = transcript.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunk = {
                        'text': current_chunk.strip(),
                        'source': source_info,
                        'chunk_id': len(chunks),
                        'type': 'audio',
                        'language': language,
                        'start_time': None,
                        'end_time': None,
                        'duration': None
                    }
                    chunks.append(chunk)
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunk = {
                'text': current_chunk.strip(),
                'source': source_info,
                'chunk_id': len(chunks),
                'type': 'audio',
                'language': language,
                'start_time': None,
                'end_time': None,
                'duration': None
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_audio(self, audio_path: str) -> List[Dict]:
        """Process audio file and return chunks"""
        print(f"🎧 Processing audio: {os.path.basename(audio_path)}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return []
        
        # Transcribe audio
        transcript_data = self.transcribe_audio(audio_path)
        
        if transcript_data['text']:
            return self.chunk_transcript(transcript_data, audio_path)
        return []
    
    def process_all_audio(self, audio_directory: str) -> List[Dict]:
        """Process all audio files in a directory"""
        all_chunks = []
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']
        
        audio_files = [f for f in os.listdir(audio_directory) 
                      if any(f.lower().endswith(fmt) for fmt in supported_formats)]
        
        if not audio_files:
            print(f"⚠️  No audio files found in {audio_directory}")
            print(f"   Supported formats: {', '.join(supported_formats)}")
            return all_chunks
        
        print(f"🎵 Found {len(audio_files)} audio files to process")
        
        for audio_file in audio_files:
            audio_path = os.path.join(audio_directory, audio_file)
            chunks = self.process_audio(audio_path)
            all_chunks.extend(chunks)
        
        print(f"✅ Processed {len(audio_files)} audio files, created {len(all_chunks)} chunks total")
        return all_chunks

# Test the audio processor
if __name__ == "__main__":
    print("🧪 Testing Audio Processor...")
    
    processor = AudioProcessor(model_size="base")
    
    # Test with data/audio directory
    audio_dir = "../data/audio"
    if os.path.exists(audio_dir):
        chunks = processor.process_all_audio(audio_dir)
        print(f"📊 Result: {len(chunks)} chunks created")
        
        if chunks:
            print("\n📝 Sample chunk:")
            sample = chunks[0]
            print(f"Source: {sample['source']}")
            print(f"Type: {sample['type']}")
            print(f"Language: {sample.get('language', 'unknown')}")
            if sample.get('start_time') is not None:
                print(f"Time: {sample['start_time']:.1f}s - {sample['end_time']:.1f}s")
            print(f"Text preview: {sample['text'][:200]}...")
    else:
        print(f"📁 Directory {audio_dir} not found. Add some audio files to test!")
        print("💡 Supported formats: .wav, .mp3, .m4a, .flac, .aac, .ogg")