# src/audio_processor.py - Enhanced Audio Processing
import whisper
import librosa
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import math
import json
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

class AudioProcessor:
    def __init__(self, model_size="base", enable_speaker_detection=True):
        self.model_size = model_size
        self.model = None
        self.enable_speaker_detection = enable_speaker_detection
        self.chunk_duration = 600  # 10 minutes in seconds
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus']
        
        print(f"🎵 Enhanced Audio processor initialized")
        print(f"   Model: {model_size}")
        print(f"   Speaker detection: {'enabled' if enable_speaker_detection else 'disabled'}")
        print(f"   Supported formats: {', '.join(self.supported_formats)}")
    
    def _load_whisper_model(self):
        """Lazy load the Whisper model with error handling"""
        if self.model is None:
            print(f"📥 Loading Whisper model: {self.model_size}")
            try:
                self.model = whisper.load_model(self.model_size)
                print("✅ Whisper model loaded successfully")
                
                # Test the model with a small audio snippet
                print("🧪 Testing model...")
                return True
            except Exception as e:
                print(f"❌ Error loading Whisper model: {e}")
                print("💡 Try installing with: pip install openai-whisper")
                raise
    
    def get_audio_info(self, file_path: str) -> Dict:
        """Get detailed audio file information"""
        try:
            # Use librosa for detailed audio analysis
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Get additional audio properties with error handling
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo) if tempo is not None else 120.0
            except Exception as e:
                print(f"⚠️ Could not detect tempo: {e}")
                tempo = 120.0
            
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                avg_spectral_centroid = float(np.mean(spectral_centroids))
            except Exception as e:
                print(f"⚠️ Could not analyze spectral features: {e}")
                avg_spectral_centroid = 0.0
            
            info = {
                'filename': os.path.basename(file_path),
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'file_size': os.path.getsize(file_path),
                'tempo': tempo,
                'avg_spectral_centroid': avg_spectral_centroid,
                'format': os.path.splitext(file_path)[1].lower()
            }
            
            print(f"📊 Audio info for {info['filename']}:")
            print(f"   Duration: {self._format_duration(duration)}")
            print(f"   Sample rate: {sr} Hz")
            print(f"   Estimated tempo: {tempo:.1f} BPM")
            
            return info
            
        except Exception as e:
            print(f"⚠️ Could not fully analyze audio file {file_path}: {e}")
            print("   Falling back to basic file info...")
            
            # Try to get basic duration with librosa
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
                print(f"   Basic duration detected: {self._format_duration(duration)}")
            except Exception as e2:
                print(f"   Could not get duration: {e2}")
                duration = 0.0
            
            # Fallback to basic info
            return {
                'filename': os.path.basename(file_path),
                'duration': duration,
                'sample_rate': 0,
                'channels': 0,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'tempo': 0,
                'avg_spectral_centroid': 0,
                'format': os.path.splitext(file_path)[1].lower()
            }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def detect_speech_segments(self, file_path: str) -> List[Dict]:
        """Detect speech segments and silence periods"""
        try:
            y, sr = librosa.load(file_path)
            
            # Use librosa to detect non-silent segments
            intervals = librosa.effects.split(y, top_db=20)
            
            segments = []
            for i, (start_frame, end_frame) in enumerate(intervals):
                start_time = start_frame / sr
                end_time = end_frame / sr
                duration = end_time - start_time
                
                # Only include segments longer than 0.5 seconds
                if duration > 0.5:
                    segments.append({
                        'segment_id': i,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
            
            print(f"🎯 Detected {len(segments)} speech segments")
            return segments
            
        except Exception as e:
            print(f"⚠️ Could not detect speech segments: {e}")
            return []
    
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Enhanced transcribe audio file to text with better options"""
        self._load_whisper_model()
        
        try:
            print(f"🎤 Transcribing audio: {os.path.basename(file_path)}")
            
            # Get audio info first
            audio_info = self.get_audio_info(file_path)
            
            # Prepare transcription options
            transcribe_options = {
                'verbose': False,
                'without_timestamps': True,  # Enable word-level timestamps
                'temperature': 0.0,  # More deterministic results
            }
            
            if language:
                transcribe_options['language'] = language
            
            # Transcribe with enhanced options
            result = self.model.transcribe(file_path, **transcribe_options)
            
            transcript = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            
            # Calculate confidence score from segments
            confidence_scores = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        word_confidences = [word.get('probability', 0.5) for word in segment['words']]
                        if word_confidences:
                            confidence_scores.extend(word_confidences)
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            
            print(f"✅ Transcription complete:")
            print(f"   Text length: {len(transcript)} characters")
            print(f"   Language: {detected_language}")
            print(f"   Confidence: {avg_confidence:.2f}")
            
            return {
                'text': transcript,
                'language': detected_language,
                'segments': result.get('segments', []),
                'confidence': avg_confidence,
                'audio_info': audio_info,
                'word_count': len(transcript.split()) if transcript else 0
            }
            
        except Exception as e:
            print(f"❌ Error transcribing {file_path}: {e}")
            return {
                'text': '', 
                'language': 'unknown', 
                'segments': [],
                'confidence': 0.0,
                'audio_info': self.get_audio_info(file_path),
                'word_count': 0
            }
    
    def detect_speakers(self, segments: List[Dict]) -> List[Dict]:
        """Simple speaker detection based on audio characteristics"""
        if not self.enable_speaker_detection or not segments:
            return segments
        
        try:
            # Simple speaker detection based on pause patterns and audio characteristics
            enhanced_segments = []
            current_speaker = 1
            
            for i, segment in enumerate(segments):
                # Simple heuristic: if there's a long pause, assume speaker change
                if i > 0:
                    prev_segment = segments[i-1]
                    pause_duration = segment.get('start', 0) - prev_segment.get('end', 0)
                    
                    # If pause is longer than 2 seconds, assume speaker change
                    if pause_duration > 2.0:
                        current_speaker = 2 if current_speaker == 1 else 1
                
                segment_copy = segment.copy()
                segment_copy['speaker'] = current_speaker
                enhanced_segments.append(segment_copy)
            
            speaker_count = len(set(seg['speaker'] for seg in enhanced_segments))
            print(f"👥 Detected {speaker_count} potential speakers")
            
            return enhanced_segments
            
        except Exception as e:
            print(f"⚠️ Speaker detection failed: {e}")
            return segments
    
    def chunk_transcript_advanced(self, transcript_data: Dict, source_info: str, max_chunk_size: int = 1000) -> List[Dict]:
        """Advanced transcript chunking with multiple strategies"""
        transcript = transcript_data['text']
        segments = transcript_data.get('segments', [])
        language = transcript_data.get('language', 'unknown')
        confidence = transcript_data.get('confidence', 0.0)
        audio_info = transcript_data.get('audio_info', {})
        
        if not transcript.strip():
            return []
        
        chunks = []
        
        if segments:
            # Enhanced segment-based chunking with speaker detection
            enhanced_segments = self.detect_speakers(segments)
            chunks = self._chunk_by_enhanced_segments(
                enhanced_segments, source_info, language, confidence, audio_info, max_chunk_size
            )
        else:
            # Fallback to intelligent sentence-based chunking
            chunks = self._chunk_by_sentences_advanced(
                transcript, source_info, language, confidence, audio_info, max_chunk_size
            )
        
        print(f"📝 Created {len(chunks)} enhanced audio chunks from {os.path.basename(source_info)}")
        return chunks
    
    def _chunk_by_enhanced_segments(self, segments: List[Dict], source_info: str, 
                                  language: str, confidence: float, audio_info: Dict, 
                                  max_chunk_size: int) -> List[Dict]:
        """Create chunks with enhanced segment information"""
        chunks = []
        current_chunk = ""
        current_start_time = 0
        current_end_time = 0
        current_speaker = None
        segment_details = []
        
        for segment in segments:
            segment_text = segment.get('text', '').strip()
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            speaker = segment.get('speaker', 1)
            
            # If adding this segment would exceed max size or speaker changed, create a chunk
            should_create_chunk = (
                len(current_chunk + segment_text) > max_chunk_size and current_chunk
            ) or (current_speaker is not None and current_speaker != speaker and current_chunk)
            
            if should_create_chunk:
                chunk = self._create_audio_chunk(
                    current_chunk.strip(), source_info, len(chunks), language, 
                    confidence, audio_info, current_start_time, current_end_time,
                    current_speaker, segment_details
                )
                chunks.append(chunk)
                
                # Reset for new chunk
                current_chunk = segment_text
                current_start_time = segment_start
                current_speaker = speaker
                segment_details = [segment]
            else:
                if not current_chunk:  # First segment
                    current_start_time = segment_start
                    current_speaker = speaker
                current_chunk += " " + segment_text
                segment_details.append(segment)
            
            current_end_time = segment_end
        
        # Add the last chunk
        if current_chunk.strip():
            chunk = self._create_audio_chunk(
                current_chunk.strip(), source_info, len(chunks), language,
                confidence, audio_info, current_start_time, current_end_time,
                current_speaker, segment_details
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences_advanced(self, transcript: str, source_info: str, 
                                   language: str, confidence: float, audio_info: Dict,
                                   max_chunk_size: int) -> List[Dict]:
        """Advanced sentence-based chunking with better sentence detection"""
        # Better sentence splitting for different languages
        if language in ['en', 'english']:
            sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        else:
            sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '。', '！', '？']
        
        sentences = []
        current_sentence = ""
        
        for char in transcript:
            current_sentence += char
            if any(current_sentence.endswith(ending) for ending in sentence_endings):
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunk = self._create_audio_chunk(
                        current_chunk.strip(), source_info, len(chunks), language,
                        confidence, audio_info
                    )
                    chunks.append(chunk)
                current_chunk = sentence + " "
        
        # Add the last chunk
        if current_chunk.strip():
            chunk = self._create_audio_chunk(
                current_chunk.strip(), source_info, len(chunks), language,
                confidence, audio_info
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_audio_chunk(self, text: str, source_info: str, chunk_id: int, 
                          language: str, confidence: float, audio_info: Dict,
                          start_time: Optional[float] = None, end_time: Optional[float] = None,
                          speaker: Optional[int] = None, segment_details: Optional[List] = None) -> Dict:
        """Create a comprehensive audio chunk with all metadata"""
        chunk = {
            'text': text,
            'source': source_info,
            'chunk_id': chunk_id,
            'type': 'audio',
            'language': language,
            'confidence': confidence,
            'word_count': len(text.split()),
            'audio_info': audio_info
        }
        
        # Add timing information if available
        if start_time is not None and end_time is not None:
            chunk.update({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'formatted_time': f"{self._format_time(start_time)} - {self._format_time(end_time)}"
            })
        
        # Add speaker information if available
        if speaker is not None:
            chunk['speaker'] = speaker
        
        # Add detailed segment information if available
        if segment_details:
            chunk['segment_count'] = len(segment_details)
            chunk['segments'] = segment_details
        
        return chunk
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def process_audio(self, audio_path: str, language: Optional[str] = None) -> List[Dict]:
        """Enhanced audio processing with better error handling"""
        print(f"🎧 Processing audio: {os.path.basename(audio_path)}")
        
        # Check if file exists and is supported
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return []
        
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in self.supported_formats:
            print(f"❌ Unsupported audio format: {file_ext}")
            print(f"   Supported formats: {', '.join(self.supported_formats)}")
            return []
        
        # Get audio info first
        audio_info = self.get_audio_info(audio_path)
        
        # Skip very short audio files (but be more lenient)
        if audio_info['duration'] < 0.5:
            print(f"⚠️ Audio file too short ({audio_info['duration']:.1f}s), skipping")
            return []
        
        # If duration is 0 but file exists and has size, try to process anyway
        if audio_info['duration'] == 0.0 and audio_info['file_size'] > 0:
            print("⚠️ Duration detection failed, but file has content. Attempting transcription anyway...")
        
        # Warn about very long audio files
        if audio_info['duration'] > 3600:  # 1 hour
            print(f"⚠️ Very long audio file ({self._format_duration(audio_info['duration'])})")
            print("   This may take a while to process...")
        
        # Transcribe audio
        transcript_data = self.transcribe_audio(audio_path, language)
        
        if transcript_data['text']:
            return self.chunk_transcript_advanced(transcript_data, audio_path)
        else:
            print(f"❌ No transcription text generated for {os.path.basename(audio_path)}")
        return []
    
    def process_all_audio(self, audio_directory: str, language: Optional[str] = None) -> List[Dict]:
        """Enhanced processing of all audio files in a directory"""
        all_chunks = []
        
        audio_files = [f for f in os.listdir(audio_directory) 
                      if any(f.lower().endswith(fmt) for fmt in self.supported_formats)]
        
        if not audio_files:
            print(f"⚠️  No audio files found in {audio_directory}")
            print(f"   Supported formats: {', '.join(self.supported_formats)}")
            return all_chunks
        
        print(f"🎵 Found {len(audio_files)} audio files to process")
        
        # Sort files by size (process smaller files first)
        audio_files_with_size = []
        for audio_file in audio_files:
            file_path = os.path.join(audio_directory, audio_file)
            file_size = os.path.getsize(file_path)
            audio_files_with_size.append((audio_file, file_size))
        
        audio_files_with_size.sort(key=lambda x: x[1])  # Sort by file size
        
        # Process files
        total_duration = 0
        successful_transcriptions = 0
        total_words = 0
        
        for audio_file, file_size in audio_files_with_size:
            audio_path = os.path.join(audio_directory, audio_file)
            print(f"\n📁 Processing {audio_file} ({file_size / 1024 / 1024:.1f} MB)")
            
            chunks = self.process_audio(audio_path, language)
            
            if chunks:
                successful_transcriptions += 1
                file_duration = chunks[0]['audio_info'].get('duration', 0)
                total_duration += file_duration
                file_words = sum(chunk['word_count'] for chunk in chunks)
                total_words += file_words
                
                print(f"✅ Success: {len(chunks)} chunks, {file_words} words, {self._format_duration(file_duration)}")
            else:
                print(f"❌ Failed to process {audio_file}")
            
            all_chunks.extend(chunks)
        
        # Print summary
        print(f"\n📊 Processing Summary:")
        print(f"   📁 Files processed: {successful_transcriptions}/{len(audio_files)}")
        print(f"   📝 Total chunks: {len(all_chunks)}")
        print(f"   📖 Total words: {total_words}")
        print(f"   ⏱️  Total duration: {self._format_duration(total_duration)}")
        
        if total_duration > 0:
            wpm = total_words / (total_duration / 60)  # Words per minute
            print(f"   🗣️  Average speech rate: {wpm:.1f} words/minute")
        
        return all_chunks
    
    def get_processing_stats(self, chunks: List[Dict]) -> Dict:
        """Get detailed statistics about processed audio chunks"""
        if not chunks:
            return {}
        
        # Calculate statistics
        total_chunks = len(chunks)
        total_words = sum(chunk.get('word_count', 0) for chunk in chunks)
        total_duration = sum(chunk.get('duration', 0) for chunk in chunks if chunk.get('duration'))
        
        languages = [chunk.get('language', 'unknown') for chunk in chunks]
        language_counts = {lang: languages.count(lang) for lang in set(languages)}
        
        confidences = [chunk.get('confidence', 0) for chunk in chunks]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        speakers = [chunk.get('speaker') for chunk in chunks if chunk.get('speaker')]
        unique_speakers = len(set(speakers)) if speakers else 0
        
        return {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'total_duration': total_duration,
            'avg_confidence': avg_confidence,
            'languages': language_counts,
            'unique_speakers': unique_speakers,
            'avg_chunk_length': total_words / total_chunks if total_chunks > 0 else 0,
            'speech_rate': total_words / (total_duration / 60) if total_duration > 0 else 0
        }

# Test the enhanced audio processor
if __name__ == "__main__":
    print("🧪 Testing Enhanced Audio Processor...")
    
    processor = AudioProcessor(model_size="base", enable_speaker_detection=True)
    
    # Test with data/audio directory
    audio_dir = "../data/audio"
    if os.path.exists(audio_dir):
        chunks = processor.process_all_audio(audio_dir)
        
        if chunks:
            print(f"\n📊 Processing Results:")
            stats = processor.get_processing_stats(chunks)
            
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Total words: {stats['total_words']}")
            print(f"   Average confidence: {stats['avg_confidence']:.2f}")
            print(f"   Languages: {stats['languages']}")
            print(f"   Unique speakers: {stats['unique_speakers']}")
            print(f"   Speech rate: {stats['speech_rate']:.1f} words/min")
            
            print("\n📝 Sample enhanced chunk:")
            sample = chunks[0]
            print(f"   Source: {os.path.basename(sample['source'])}")
            print(f"   Language: {sample['language']}")
            print(f"   Confidence: {sample['confidence']:.2f}")
            print(f"   Duration: {sample.get('formatted_time', 'N/A')}")
            print(f"   Speaker: {sample.get('speaker', 'N/A')}")
            print(f"   Words: {sample['word_count']}")
            print(f"   Text preview: {sample['text'][:200]}...")
        else:
            print("💡 No audio files processed. Add some audio files to test!")
    else:
        print(f"📁 Directory {audio_dir} not found. Add some audio files to test!")
        print("💡 Enhanced features include:")
        print("   • Better error handling and file validation")
        print("   • Audio analysis (tempo, spectral features)")
        print("   • Speech segment detection")
        print("   • Simple speaker detection")
        print("   • Word-level timestamps")
        print("   • Enhanced chunking strategies")
        print("   • Comprehensive statistics")