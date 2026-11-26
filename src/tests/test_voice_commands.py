import io,json,unittest
from unittest.mock import patch
import numpy as np
from voice_commands import SpeechChunks,transcribe,VoiceCommands
import queue,time

class Tests(unittest.TestCase):
 def test_silence_and_short_noise_ignored(self):
  v=SpeechChunks();quiet=np.zeros(1600,dtype='<i2').tobytes();loud=np.full(1600,1000,dtype='<i2').tobytes()
  for _ in range(12):self.assertIsNone(v.feed(quiet))
  self.assertIsNone(v.feed(loud))
  for _ in range(10):self.assertIsNone(v.feed(quiet))
 def test_speech_endpoint(self):
  v=SpeechChunks();loud=np.full(1600,1000,dtype='<i2').tobytes();quiet=np.zeros(1600,dtype='<i2').tobytes()
  for _ in range(5):self.assertIsNone(v.feed(loud))
  for _ in range(7):self.assertIsNone(v.feed(quiet))
  self.assertIsNotNone(v.feed(quiet))
 def test_api_schema_and_explicit_command(self):
  def respond(req,timeout):
   self.assertIn(b'name="file"',req.data);self.assertIn(b'RIFF',req.data)
   self.assertEqual(timeout,20)
   return io.BytesIO(json.dumps({'text':'Pick and place the matchbox.'}).encode())
  with patch.dict('os.environ',{'OPENAI_API_KEY':'test-key'}),patch('urllib.request.urlopen',side_effect=respond):
   self.assertEqual(transcribe(b'\0'*32000),'pick and place matchbox')
  with patch.dict('os.environ',{'OPENAI_API_KEY':'test-key'}),patch('urllib.request.urlopen',return_value=io.BytesIO(b'{"text":"Do not pick and place the matchbox"}')):
   with self.assertRaises(ValueError):transcribe(b'\0'*32000)
 def test_old_transcripts_do_not_execute_after_mute(self):
  v=VoiceCommands.__new__(VoiceCommands);v.enabled=True;v.epoch=3;v.results=queue.Queue();v.status=''
  v.results.put((2,time.monotonic(),'pick and place box'));self.assertIsNone(v.poll())
  v.results.put((3,time.monotonic()-3,'pick and place box'));self.assertIsNone(v.poll())
  v.results.put((3,time.monotonic(),'pick and place box'));self.assertEqual(v.poll(),'pick and place box')

if __name__=='__main__':unittest.main()
