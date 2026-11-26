"""Create a short plain-language PDF and text companion. No hardware access."""
from pathlib import Path
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT=Path(__file__).resolve().parents[1]/'reports/setup_explained'
pages=[
('1. The setup and what OpenCV does',[
('SO-101 MJCF model — the virtual robot',
'MJCF is MuJoCo’s XML model format. It describes the robot’s parts, shapes, joints and how they connect. We reused an existing SO-101 model instead of reconstructing its mechanics from photographs. In your setup, this is the white virtual arm; the desk and approximate camera holder were added around it. File: assets/so101/so101.xml.'),
('Measured joint feedback — where the real arm actually is',
'The motors report their encoder positions. We convert those readings into the virtual robot’s joint angles, including the elbow, wrist-bend and wrist-roll offsets we adjusted together. For example, bending the real elbow updates the simulated elbow. This is measured position, distinct from the position we ask the motor to reach. Code: command_mapping.Mapping.'),
('Overhead and wrist cameras — two different jobs',
'The fixed overhead camera observes the table so the program can estimate object and plate locations. The wrist camera gives a close view near the jaws. We attempted to calibrate its optical mounting, but validation was poor; it currently helps you observe the grasp rather than supplying trusted metric localization.'),
('Checkerboard calibration — turn pixels into useful geometry',
'OpenCV finds the printed checkerboard corners and estimates focal lengths, image centre and lens distortion. Your printed 100 mm bar measured 96 mm, so we used 24 mm squares instead of nominal 25 mm squares. This corrected the metric scale. Camera pose relative to the table and arm required separate registration. Functions: cv2.findChessboardCornersSB(), cv2.calibrateCamera().'),
('Fiducial detection — recognize a known reference marker',
'A fiducial is a deliberately designed visual marker with identifiable corners. We temporarily attached an ArUco marker to the fixed gripper structure and observed it at different arm poses. Its detected corners helped fit the camera-to-robot relationship. This is what the printed gripper marker was for. Function: cv2.aruco.ArucoDetector.'),
('Image differencing — notice something changed',
'The program compares the current table image with a saved baseline. Persistent color changes suggest an added or moved object. For example, adding the second box produces a new changed region. Color compensation and robot masking reduce false changes from lighting or arm movement. Code: scene_changes.Changes.'),
('Contour geometry — estimate shape, width and direction',
'A contour is the outline of an image region. We project it onto the calibrated table plane and estimate a rectangle. For your small box, this provides an approximate width and long-axis direction for opening and orienting the jaws. Shadows or tall objects can bias this estimate. Code: scene_changes.Plane.'),
('Destination matching — find the plate again',
'When you move the yellow plate, the program searches the new image using reference features, color/ellipse geometry and template matching. It updates the destination before planning rather than deliberately aiming at its old coordinates. Failed detection stops planning. Code: basket_tracking.measure_basket().'),
]),
('2. What MuJoCo and SciPy do',[
('MuJoCo geometry — describe where solid parts are',
'MuJoCo stores the robot’s link shapes and their positions. We also use a table plane and simple object proxies. For example, before descending toward the packet, the planner checks sampled arm and holder geometry against the table. These are approximate checks, not complete protection against cables, self-collision or every tabletop obstacle.'),
('Forward kinematics — angles in, fingertip position out',
'Given shoulder, elbow and wrist angles, forward kinematics calculates where the gripper ends up. For example: “With the elbow bent this much, how high are the fingertips?” MuJoCo evaluates the model; our ReachSolver.position() retrieves the gripper reference position using mujoco.mj_forward().'),
('Rendering — draw the predicted pose',
'Rendering turns the three-dimensional model into an image. This produces the virtual arm view and the overlay beside your overhead image. It lets you inspect whether the simulated holder or gripper is aligned and preview a proposed reach. A convincing rendered motion does not prove that the physical arm completed it.'),
('Separate contact experiments — test virtual touching and lifting',
'In contact_preview.py, a simulated highlighter is a free object that can interact with the fingers and table. We tested closing, lifting and releasing it using MuJoCo contact physics. These experiments used assumed mass and friction. They differ from the ordinary real-arm planner, which assumes the carried object stays in the grasp.'),
('SciPy bounded nonlinear optimization — find usable joint angles',
'Inverse kinematics asks the reverse question: “Which joint angles put the jaws over this box?” scipy.optimize.least_squares() repeatedly adjusts candidate angles to reduce position and orientation errors. “Bounded” means the search respects joint-angle limits. “Nonlinear” means fingertip position changes through coupled rotations, rather than a simple straight-line formula.'),
('Optimization example from your setup',
'When you moved the object, ReachSolver.solve_natural_grasp() searched for a new reachable grasp pose. Trying alternative wrist-roll orientations helped when one direction had already rolled too far. The solver computes angles from geometry; it does not learn them from previous grasps. It may still fail if the requested pose cannot satisfy the constraints.'),
('Scalar root finding — choose one gripper angle for a gap',
'Here “scalar” means one unknown: the gripper hinge angle. scipy.optimize.brentq() searches for an angle where calculated jaw gap minus requested gap equals zero. For example, a 50 mm opening request becomes a model hinge angle. The restored second-box rule asks for estimated width minus 4 mm: a 43 mm box gives 39 mm. This commands position, not a measured grip force.'),
]),
('3. LeRobot, AI and one complete example',[
('LeRobot motor interface — communicate with the real arm',
'LeRobot provides the SO101Follower interface and motor-bus access. Our robot_setup.make_robot() creates it, and control_worker.Controller handles the physical command loop. For example, the planner produces a sequence of joint targets; the worker uses the motor interface to send them and read the resulting positions. The viewer itself is not a separate competing motor controller.'),
('Calibration conventions — agree on what a motor reading means',
'LeRobot calibration records reference offsets and travel ranges so raw encoder values can be interpreted consistently. This was the step where you placed the arm near midrange and swept the joints by hand. Our project then added a separate motor-to-MuJoCo mapping. Motor calibration and the later visual elbow/wrist adjustments are related, but are not the same operation.'),
('Classical geometry plus optional GPT recognition',
'Object outlines, tracking and plate localization use explicit image-change, color and geometry methods. The new GPT mode adds a pretrained vision-language model to identify a requested object among those outlines. Pages 4–7 explain how recognition connects to adaptive tracking and physical grasp planning.'),
('No reinforcement-learning policy',
'A reinforcement-learning policy would learn actions from rewards during trials or simulation. Our arm instead follows explicitly constructed reach, descend, close, transfer and open stages, using inverse kinematics and timed joint paths. The successful grasp experiments did not automatically train a policy.'),
('GPT selects identity; geometry and feedback control motion',
'The optional runtime now sends an overhead image and your object description to GPT. It returns a candidate object ID. OpenCV supplies its current footprint, SciPy and MuJoCo calculate the motion, and LeRobot communicates with the motors. GPT does not directly command the arm. This is separate from the AI assistant used to develop the software.'),
('Example: put the small box in the plate',
'1. OpenCV finds the box footprint and the plate in a fresh overhead frame.\n2. The calibrated geometry expresses their approximate positions relative to the robot.\n3. SciPy searches for joint angles; MuJoCo calculates and displays the resulting poses.\n4. A root solve converts the requested jaw gap into a gripper angle.\n5. Our controller generates timed commands, and the LeRobot interface sends them to the motors.\n6. Encoder feedback reports how the arm actually moved; the wrist view helps you inspect the grasp.'),
('Scope of this explanation',
'This describes the inspected project implementation, including the restored second-object grip setting. Object heights remain supplied assumptions, wrist optical registration remains unaccepted, and physical grasp success is not guaranteed by a successful simulation preview. Source files and detailed evidence are indexed in the main project report.'),
])]

from report_extensions import PAGES
pages.extend(PAGES)

plain=['YOUR SO-101 SETUP — THE TOOLS EXPLAINED','A short companion to the project report. Examples refer to our actual implementation.','']
for title,sections in pages:
 plain += [title,'='*len(title),'']
 for heading,body in sections:plain += [heading,body,'']
(OUT/'Setup_Tools_Explained.txt').write_text('\n'.join(plain),encoding='utf-8')
fonts='/usr/share/fonts/truetype/dejavu/'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,fonts+file))
navy=colors.HexColor('#163348');teal=colors.HexColor('#087F8C')
title_style=ParagraphStyle('title',fontName='Bold',fontSize=17,leading=22,textColor=navy,spaceAfter=12)
head=ParagraphStyle('head',fontName='Bold',fontSize=10.2,leading=13,textColor=teal,spaceBefore=8,spaceAfter=3,keepWithNext=True)
body_style=ParagraphStyle('body',fontName='Body',fontSize=9.3,leading=12.7,textColor=navy,spaceAfter=3)
W,H=A4

def footer(c,doc):
 c.saveState();c.setFillColor(colors.HexColor('#5b6b75'));c.setFont('Body',8)
 c.drawString(43,25,'SOARM–MuJoCoMirror | Plain-language companion | 20 September 2026')
 c.drawRightString(W-43,25,str(doc.page));c.restoreState()

def pipeline_diagram():
 from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
 d=Drawing(505,106)
 labels=[['Camera + contours','GPT selects ID','Live metric footprint','Grasp IK + checks'],['Open / prepare','Adaptive reach','Descend','Close + hold']]
 for row,items in enumerate(labels):
  y=70-row*53
  for col,label in enumerate(items):
   x=col*128
   d.add(Rect(x,y,118,29,rx=4,ry=4,fillColor=colors.HexColor('#eef5f6'),strokeColor=teal))
   d.add(String(x+59,y+11,label,fontName='Body',fontSize=7.3,textAnchor='middle',fillColor=navy))
   if col<3:
    d.add(Line(x+118,y+14,x+126,y+14,strokeColor=teal))
    d.add(Polygon([x+126,y+14,x+122,y+17,x+122,y+11],fillColor=teal,strokeColor=teal))
 d.add(String(252,56,'G starts execution after selection and planning',fontName='Body',fontSize=7.2,textAnchor='middle',fillColor=navy))
 d.add(String(252,3,'Live overhead tracking + encoder feedback accompany the motion stages',fontName='Body',fontSize=7.2,textAnchor='middle',fillColor=navy))
 return d

story=[]
for i,(title,sections) in enumerate(pages):
 if i:story.append(PageBreak())
 story.append(Paragraph(escape(title),title_style))
 if title.startswith('6.'):story.extend([pipeline_diagram(),Spacer(1,8)])
 if not i:story.append(Paragraph('Your setup, explained through concrete examples.',body_style))
 for heading,body in sections:
  story.append(Paragraph(escape(heading),head))
  story.append(Paragraph(escape(body).replace('\n','<br/>'),body_style))
doc=SimpleDocTemplate(str(OUT/'Setup_Tools_Explained.pdf'),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=35,bottomMargin=42,title='Your SO-101 Setup: The Tools Explained',author='Prepared for Saeed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Created PDF and text:',OUT)
