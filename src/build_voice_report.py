from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Flowable
from reportlab.lib.enums import TA_CENTER
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]/'reports/voice_pick_place'
for name,file in [('Arial','Arial.TTF'),('Arial-Bold','Arialbd.TTF'),('Arial-Italic','Ariali.TTF')]:
    pdfmetrics.registerFont(TTFont(name,str(ROOT/'fonts'/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
styles={
 'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#15384b'),spaceAfter=13),
 'sub':ParagraphStyle('sub',fontName='Arial',fontSize=10,leading=15,textColor=colors.HexColor('#506572'),spaceAfter=16),
 'h':ParagraphStyle('h',fontName='Arial-Bold',fontSize=13,leading=17,textColor=colors.HexColor('#15384b'),spaceBefore=13,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10.5,leading=15.5,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=9,leading=13,spaceAfter=7),
 'box':ParagraphStyle('box',fontName='Arial',fontSize=10,leading=14,alignment=TA_CENTER),
}
story=[];text=[]
def p(s,kind='body'):
    story.append(Paragraph(s,styles[kind]));text.append(s)
def title(s,sub):p(s,'title');p(sub,'sub')
def section(s):p(s,'h')
def page():story.append(PageBreak())
class Pipeline(Flowable):
    def __init__(self):super().__init__();self.width=475;self.height=240
    def draw(self):
        labels=[('Your voice + real overhead image','Audio and images are the inputs'),('Speech API + OpenCV','Transcribe command; find and number objects'),('GPT visual selection','Return the requested object’s candidate ID'),('Calibrated geometry + SciPy + MuJoCo','Measure target and plate; plan all stages'),('Controller + LeRobot + real arm','Send joint commands; read motor feedback')]
        c=self.canv
        for i,(a,b) in enumerate(labels):
            y=240-(i+1)*47
            c.setFillColor(colors.HexColor('#edf4f7'));c.setStrokeColor(colors.HexColor('#bbd0da'));c.roundRect(0,y,475,38,5,fill=1,stroke=1)
            c.setFillColor(colors.HexColor('#15384b'));c.setFont('Arial-Bold',10);c.drawCentredString(237.5,y+23,a)
            c.setFont('Arial',9);c.drawCentredString(237.5,y+9,b)
            if i<4:
                c.line(237.5,y,237.5,y-8);c.line(237.5,y-8,234.5,y-5);c.line(237.5,y-8,240.5,y-5)

title('From voice to pick-and-place','SO-101 / MuJoCo setup • Current fixed-plan pipeline • 21 September 2026')
p('Example command: <b>“Pick and place the biscuit.”</b> GPT selects the object. Our Python application connects that selection to measured geometry, motion planning, and the real motor controller.')
story.append(Pipeline());story.append(Spacer(1,5))
section('1. Capture speech and extract the command')
p('<b>voice_commands.py</b> records microphone audio using Linux <b>arecord</b>. A simple audio-energy and silence check separates a spoken sentence from background silence; this part is not a neural speech detector.')
p('The audio is sent to OpenAI’s transcription API. The configured default model, <b>gpt-4o-mini-transcribe</b>, returns text such as “Pick and place the biscuit.”')
p('<b>parse_pick_place_command()</b> runs locally. It recognizes “pick and place” as execution intent and extracts “biscuit” as the object description. A valid explicit command starts recognition, planning, and execution without pressing A, P, or G.')
p('The microphone listens while the application is idle and pauses during movement. Space stops motion; X releases torque. Voice is not the emergency-stop mechanism.','small')
page()
title('Finding and identifying the object','OpenCV measures candidate shapes; GPT supplies their semantic identity.')
section('2. Detect possible objects with OpenCV')
p('The input is the <b>real fixed overhead camera image</b>, not a MuJoCo rendering. The application corrects lens distortion using our camera calibration.')
p('<b>Changes.update()</b> compares the current image with a saved tabletop baseline in blurred LAB color space. It compensates approximately for image-wide lighting changes, thresholds the differences, cleans the mask using morphological operations, and extracts outlines with <b>cv2.findContours()</b>.')
p('Further contour refinement reduces shadows and fragmented detections. A projected robot mask excludes the arm from candidate detection. These are classical image-processing rules, not a trained object detector. They produce possible object shapes without knowing which is a biscuit.')
section('3. Ask GPT which candidate matches your request')
p('<b>gpt_targets.request_payload()</b> draws numbered outlines around the detected candidates. It sends the image and the object description “biscuit” to the OpenAI Responses API. The configured default vision model is <b>gpt-4.1-mini</b>.')
p('GPT uses appearance, packaging, and visible text to choose among the candidates. In the recorded biscuit attempt, it selected <b>candidate 2</b> and described a green-and-yellow packet. The response is structured JSON containing <b>status</b>, <b>candidate_id</b>, and <b>description</b>; for example, status “selected” and candidate_id 2.')
p('<b>The response goes back to the Python application.</b> GPT does not communicate with the motors or provide joint angles. <b>parse_selection()</b> checks the response and rejects invalid IDs, absent targets, and ambiguous selections. Candidates are tracked while the API responds, and the selected one must still be stable.')
section('4. Turn the selected outline into robot coordinates')
p('Candidate 2 already has an OpenCV contour. <b>Plane.footprint()</b> projects its pixels onto the calibrated tabletop plane and transforms those coordinates into the robot base frame. <b>cv2.minAreaRect()</b>, applied in metric tabletop coordinates, estimates center, width, length, and orientation.')
p('Object height comes from our configured measurements, not automatic depth estimation or GPT. Existing grasp offsets compensate for the alignment differences observed during physical testing. These measurements remain approximations, especially for raised or irregular objects.')
page()
title('Planning the complete movement','SciPy searches for joint angles; MuJoCo evaluates the resulting geometry.')
section('5. Measure the plate independently')
p('<b>current_plate()</b> measures the destination again from the planning image. The detector uses reference-image feature matching, with a colored-rim and ellipse fallback. Camera calibration converts the detected opening into a physical destination. The plate’s configured rim height is <b>2 cm</b>.')
p('The destination is part of our programmed routine; GPT does not currently choose it. The detector uses the current image rather than silently reusing an old plate location.')
section('6. Solve and check the full pick-and-place sequence')
p('The planner uses the selected object’s center, orientation, estimated width, configured height, and grasp offsets to define desired gripper poses. It starts from measured robot joint feedback.')
p('<b>SciPy</b> performs bounded nonlinear optimization to find suitable joint angles. For each candidate configuration, <b>MuJoCo</b> computes forward kinematics: where the gripper and robot geometry would be. The planner evaluates position error, orientation, joint limits, and modeled table/plate clearances.')
p('The planned sequence is:<br/><b>Prepare/open → reach above object → descend → close → lift → move over plate → open → return to sitting.</b>')
p('<b>build_sequence()</b> prepares the approach. <b>build_pick_place_tail()</b> combines <b>build_tail()</b> for descent and closure with <b>append_basket()</b> for lifting, plate placement, and return. “Basket” is a historical internal name; the current destination is the plate.')
p('The closing gap is based on estimated object width and our configured offsets. SciPy’s scalar root finder, <b>brentq()</b>, finds the gripper joint angle that produces the requested modeled jaw gap. The biscuit/cracker command now requests an additional <b>5 mm smaller gap</b>; the minimum commanded gap remains 5 mm.')
p('The entire sequence is checked before execution. MuJoCo supports geometry, kinematics, clearance evaluation, and visualization. This operational planner does <b>not</b> simulate the exact friction, packet deformation, or grasp force. A planned small jaw gap is not a calibrated force command.')
section('What information leaves the computer?')
p('The speech clip goes to the transcription API. The numbered overhead image and object description go to the vision API. Contour measurement, camera-to-robot mapping, motion planning, and motor control run locally. The API response selects a candidate; our local software determines its motion.')
page()
title('Executing on the real arm','Motor feedback closes the joint-control loop; the target path stays fixed.')
section('7. Send the planned trajectory to the motors')
p('<b>adaptive_grip.py</b> passes planned segments to the separate <b>control_worker.py</b> process. The worker interpolates each segment over time and uses our calibrated mapping to convert joint angles into raw servo position units.')
p('Through <b>LeRobot’s motor-bus interface</b>, the worker writes <b>Goal_Position</b> commands to the six motors and reads <b>Present_Position</b> feedback. The application checks endpoint errors before advancing to the next stage. It also monitors controller communication and feedback availability.')
p('During grasping and carrying, a small jaw-position difference is allowed because the object can prevent full commanded closure. In the biscuit attempt, the arm reached the target, but the jaw remained about 5.6° short of its command. The old 5° endpoint tolerance stopped the sequence; the holding-stage jaw tolerance was adjusted to 6°. Arm-position checks were preserved.')
section('8. Continue through camera occlusion')
p('Once execution begins, the system follows the <b>complete fixed plan</b>. The arm can block the overhead view of the object without forcing another detection or stopping the sequence. Motor feedback is still checked, but object position is not continuously updated during this movement.')
p('This mode assumes the object and plate stay in place after planning. It does not currently use the wrist camera to correct the final approach. “Placement complete” means the commanded sequence finished; it does not independently verify that the object was securely lifted or successfully released.')
section('Where AI is used—and what “policy” means here')
p('<b>GPT Astra helped develop the software.</b> The runtime AI components are speech transcription and GPT visual object identification. The model names above are code defaults and can be changed by configuration.')
p('The manipulation policy is a programmed perception–planning–control pipeline. There is no trained neural motor policy or reinforcement-learning controller in this path. OpenCV supplies classical image processing; SciPy supplies numerical solvers; MuJoCo supplies robot geometry and kinematics; LeRobot supplies the motor interface.')
section('Code reference')
p('Voice: voice_commands.py, pick_place_command.py<br/>Identification: gpt_targets.py<br/>Geometry: scene_changes.py, plate_detection.py, basket_tracking.py<br/>Planning: mirror_grasp_plan.py, grasp_tail.py, gpt_pick_place.py, basket_place.py<br/>Execution: adaptive_grip.py, control_worker.py, command_mapping.py, replay_settling.py','small')

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#c9d6dd'));c.line(48,43,547,43)
 c.setFont('Arial',8);c.setFillColor(colors.HexColor('#506572'));c.drawString(48,29,'SO-101 • Voice-controlled pick-and-place');c.drawRightString(547,29,str(doc.page))
out=ROOT/'Voice_to_Pick_and_Place_Pipeline.pdf'
doc=SimpleDocTemplate(str(out),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=47,bottomMargin=57,title='Voice to Pick-and-Place: SO-101 Pipeline',author='Saeed',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'pipeline_text.txt').write_text('\n\n'.join(text))
print(out)
