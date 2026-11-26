"""Idle-time microphone commands via ALSA and OpenAI transcription. No robot I/O."""
import collections,io,json,os,queue,subprocess,threading,time,urllib.request,urllib.error,uuid,wave
import numpy as np
from pick_place_command import parse_pick_place_command


def transcribe(pcm):
    key=os.environ.get('OPENAI_API_KEY')
    if not key:raise ValueError('OPENAI_API_KEY is not configured')
    audio=io.BytesIO()
    with wave.open(audio,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
    boundary='soarm-'+uuid.uuid4().hex;parts=[]
    for name,value in [('model',os.environ.get('OPENAI_TRANSCRIBE_MODEL','gpt-4o-mini-transcribe')),('language','en'),('response_format','json')]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="command.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),audio.getvalue(),f'\r\n--{boundary}--\r\n'.encode()]
    req=urllib.request.Request('https://api.openai.com/v1/audio/transcriptions',data=b''.join(parts),headers={'Authorization':'Bearer '+key,'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=20) as response:result=json.load(response)
    except urllib.error.HTTPError as e:raise ValueError(f'Transcription API error {e.code}') from None
    except (OSError,urllib.error.URLError):raise ValueError('Transcription connection failed') from None
    text=result.get('text','')
    if not isinstance(text,str) or len(text)>100:raise ValueError('Voice command too long or invalid')
    execute,target=parse_pick_place_command(text)
    if not execute:raise ValueError('Not a pick-and-place command: '+text[:70])
    return 'pick and place '+target


class SpeechChunks:
    """Simple energy endpointing of 100 ms mono PCM chunks (not neural VAD)."""
    def __init__(self):self.reset()
    def reset(self):self.pre=collections.deque(maxlen=3);self.chunks=[];self.silent=0;self.voiced=0
    def feed(self,chunk):
        a=np.frombuffer(chunk,dtype='<i2').astype(float)
        loud=len(a)>0 and np.sqrt(np.mean(a*a))>=float(os.environ.get('SOARM_VOICE_RMS','350'))
        if not self.chunks:
            self.pre.append(chunk)
            if loud:self.chunks=list(self.pre);self.voiced=1
            return None
        self.chunks.append(chunk);self.voiced+=int(loud);self.silent=0 if loud else self.silent+1
        if len(self.chunks)>=80:
            self.reset();return None # reject prolonged/noisy speech, never truncate into a command
        if self.silent>=8:
            out=b''.join(self.chunks) if self.voiced>=4 else None
            self.reset();return out
        return None


class VoiceCommands:
    def __init__(self):
        self.enabled=False;self.epoch=0;self.busy=False;self.status='Microphone paused';self.results=queue.Queue();self.stop=threading.Event()
        self.process=subprocess.Popen(['arecord','-q','-D',os.environ.get('SOARM_MIC_DEVICE','default'),'-t','raw','-f','S16_LE','-r','16000','-c','1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        self.thread=threading.Thread(target=self._capture,daemon=True);self.thread.start()
    def listen(self,enabled):
        if enabled!=self.enabled:
            self.enabled=enabled;self.epoch+=1
            self.status='Listening: say pick and place ...' if enabled else 'Microphone paused'
    def _capture(self):
        vad=SpeechChunks();epoch=self.epoch
        while not self.stop.is_set():
            chunk=self.process.stdout.read(3200)
            if len(chunk)!=3200:
                self.status='Microphone unavailable';return
            if epoch!=self.epoch:vad.reset();epoch=self.epoch
            if not self.enabled or self.busy:vad.reset();continue
            pcm=vad.feed(chunk)
            if pcm:
                self.busy=True;self.status='Transcribing voice command...'
                def work(data,token):
                    try:result=transcribe(data)
                    except Exception as exc:result=exc
                    self.results.put((token,time.monotonic(),result));self.busy=False
                threading.Thread(target=work,args=(pcm,epoch),daemon=True).start()
    def poll(self):
        try:token,finished,result=self.results.get_nowait()
        except queue.Empty:return None
        if not self.enabled or token!=self.epoch or time.monotonic()-finished>2:return None
        if isinstance(result,Exception):self.status=str(result);return None
        self.status='Heard: '+result;return result
    def close(self):
        self.stop.set();self.enabled=False;self.epoch+=1
        self.process.terminate()
        try:self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self.thread.join(timeout=1);self.process.stdout.close()
