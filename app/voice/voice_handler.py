import asyncio
import edge_tts
import whisper
import tempfile
import os
import subprocess
import io
import wave
import numpy as np
import speech_recognition as sr


class VoiceHandler:
    def __init__(
        self,
        whisper_model: str = "large",
        listen_timeout: int = 10,
        phrase_time_limit: int | None = None,
        pause_threshold: float = 0.9,
    ):
        """
        Initialize voice handler with Whisper for STT and edge-tts for TTS.
        
        Args:
            whisper_model: "tiny", "base", "small", "medium", or "large"
            listen_timeout: Seconds to wait for speech to start.
            phrase_time_limit: Maximum seconds to record once speech starts.
                Set to None to stop naturally when you pause.
            pause_threshold: Silence seconds that indicate you're done speaking.
        """
        print(f"Loading Whisper model: {whisper_model}...")
        self.whisper_model = whisper.load_model(whisper_model)
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = pause_threshold
        self.listen_timeout = listen_timeout
        self.phrase_time_limit = phrase_time_limit

    def _play_mp3_without_vlc_window(self, audio_path: str) -> None:
        """Play mp3 in-process via PowerShell MediaPlayer to avoid VLC/manual close."""
        escaped_path = audio_path.replace("'", "''")
        play_script = (
            "Add-Type -AssemblyName PresentationCore; "
            f"$path = '{escaped_path}'; "
            "$player = New-Object System.Windows.Media.MediaPlayer; "
            "$player.Open([Uri]$path); "
            "$player.Play(); "
            "$attempts = 0; "
            "while (-not $player.NaturalDuration.HasTimeSpan -and $attempts -lt 200) "
            "{ Start-Sleep -Milliseconds 50; $attempts++ }; "
            "if ($player.NaturalDuration.HasTimeSpan) "
            "{ Start-Sleep -Milliseconds ([int]$player.NaturalDuration.TimeSpan.TotalMilliseconds + 100) } "
            "else { Start-Sleep -Seconds 2 }; "
            "$player.Stop(); "
            "$player.Close();"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", play_script],
            check=False,
        )

    def _audio_data_to_whisper_input(self, audio: sr.AudioData) -> np.ndarray:
        """Convert microphone audio to 16 kHz float32 mono for Whisper."""
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(wav_file.getnframes())

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples

    async def _speak_edge_tts(self, text: str) -> None:
        """Use edge-tts to speak text out loud."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_file = f.name

        try:
            communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
            await communicate.save(temp_file)
            self._play_mp3_without_vlc_window(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def speak(self, text: str) -> None:
        """Speak text out loud (main entry point)."""
        asyncio.run(self._speak_edge_tts(text))

    def listen(
        self,
        prompt: str = "",
        timeout: int | None = None,
        phrase_time_limit: int | None = None,
    ) -> str:
        """
        Speak a prompt and listen for the user's voice response.
        Returns the recognized text.
        """
        if prompt:
            self.speak(prompt)

        print("[Listening...]")
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self.recognizer.listen(
                    source,
                    timeout=self.listen_timeout if timeout is None else timeout,
                    phrase_time_limit=(
                        self.phrase_time_limit
                        if phrase_time_limit is None
                        else phrase_time_limit
                    ),
                )
            
            whisper_audio = self._audio_data_to_whisper_input(audio)
            result = self.whisper_model.transcribe(
                whisper_audio,
                fp16=False,
                language="en",
                temperature=0.0,
                beam_size=5,
                best_of=5,
            )
            recognized_text = result["text"].strip()
            
            print(f"[You said: {recognized_text}]")
            return recognized_text

        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
            return ""
        except Exception as e:
            print(f"Error during listening: {e}")
            return ""
