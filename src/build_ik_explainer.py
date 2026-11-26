"""Offline explanatory sketches and PDF; no hardware access."""
from pathlib import Path
import math, html
from PIL import Image,ImageDraw,ImageFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Image as PDFImage,PageBreak,Spacer,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
P=Path(__file__).resolve().parents[1]/'reports/ik_explained'
F='/usr/share/fonts/truetype/dejavu/'
navy='#17374b'; teal='#087f8c'; orange='#df851c'; gray='#657985'
def font(n,b=False):return ImageFont.truetype(F+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),n)
def arrow(d,a,b,color=teal,width=5):
 d.line([a,b],fill=color,width=width);ang=math.atan2(b[1]-a[1],b[0]-a[0]);s=18
 d.polygon([b,(b[0]-s*math.cos(ang-.45),b[1]-s*math.sin(ang-.45)),(b[0]-s*math.cos(ang+.45),b[1]-s*math.sin(ang+.45))],fill=color)
def text(d,pos,s,size=25,color=navy,bold=False):d.multiline_text(pos,s,font=font(size,bold),fill=color,spacing=8)
def chain(d,points,color=teal,width=16):
 d.line(points,fill=color,width=width)
 for x,y in points[:-1]:d.ellipse((x-13,y-13,x+13,y+13),fill='white',outline=color,width=5)
 x,y=points[-1];d.line([(x-20,y),(x+20,y)],fill=color,width=6);d.line([(x-20,y),(x-20,y+30)],fill=color,width=6);d.line([(x+20,y),(x+20,y+30)],fill=color,width=6)

im=Image.new('RGB',(1600,670),'#f4f8fa');d=ImageDraw.Draw(im)
text(d,(55,30),'FORWARD KINEMATICS',30,bold=True);text(d,(850,30),'INVERSE KINEMATICS',30,bold=True)
text(d,(55,80),'Known: joint angles',25);text(d,(850,80),'Known: desired gripper pose',25)
d.line((800,30,800,630),fill='#cfdae0',width=3)
for ox in (0,800):d.line((ox+50,540,ox+750,540),fill=gray,width=4);text(d,(ox+60,555),'Table (side-view sketch)',21,color=gray)
chain(d,[(170,535),(270,270),(530,350),(595,420)])
text(d,(110,430),'shoulder',22);text(d,(195,210),'elbow',22);text(d,(455,292),'wrist',22)
d.ellipse((586,411,604,429),fill=orange);text(d,(360,470),'Calculated gripper position',22,color=orange)
chain(d,[(970,535),(1060,260),(1310,335),(1390,395)],'#bac9d2',11)
chain(d,[(970,535),(1160,315),(1410,355),(1410,460)])
d.rectangle((1385,505,1435,539),fill='#ddb058');d.ellipse((1400,450,1420,470),fill=orange)
text(d,(1190,570),'Box',22,color=orange);text(d,(980,150),'Search for a new set of angles',24,color=teal)
text(d,(55,615),'Angles → calculated pose',26,bold=True);text(d,(850,615),'Desired pose → possible angles',26,bold=True)
im.save(P/'fk_ik.png')

im=Image.new('RGB',(1600,780),'#f4f8fa');d=ImageDraw.Draw(im)
boxes=[(55,65,460,180,'1  SciPy proposes angles','Shoulder, elbow, wrist…'),(960,65,580,180,'2  MuJoCo evaluates pose','mj_forward() + model geometry'),(960,455,580,210,'3  Our Python calculates error','Position mismatch + orientation\nmismatch; return residual to SciPy'),(55,455,460,210,'4  SciPy improves the guess','least_squares() chooses an update\nwithin the joint-angle bounds')]
for x,y,w,h,title,sub in boxes:
 d.rounded_rectangle((x,y,x+w,y+h),20,fill='white',outline=teal,width=4);text(d,(x+22,y+25),title,26,bold=True);text(d,(x+22,y+88),sub,23)
arrow(d,(515,150),(960,150));arrow(d,(1250,245),(1250,455));arrow(d,(960,555),(515,555));arrow(d,(285,455),(285,245))
text(d,(560,300),'Repeat in software',29,bold=True);text(d,(560,352),'No physical trial motion\nis needed for these guesses.',23,color=gray)
text(d,(70,711),'After acceptance: build path → check sampled geometry → preview → execute',26,bold=True)
im.save(P/'solver_loop.png')

im=Image.new('RGB',(1600,530),'#f4f8fa');d=ImageDraw.Draw(im)
text(d,(45,25),'WHY POSITION ALONE IS NOT ENOUGH',30,bold=True)
for x,label,correct in [(80,'Wrong jaw orientation',False),(850,'Useful jaw orientation',True)]:
 d.rectangle((x+170,190,x+470,300),fill='#d6b778',outline=orange,width=3)
 cx=x+320;cy=245
 if correct:
  d.line((x+160,145,x+480,145),fill=teal,width=14);d.line((x+160,345,x+480,345),fill=teal,width=14)
  arrow(d,(cx,155),(cx,185));arrow(d,(cx,335),(cx,305))
 else:
  d.line((x+120,180,x+120,310),fill=gray,width=14);d.line((x+520,180,x+520,310),fill=gray,width=14)
  arrow(d,(x+135,cy),(x+165,cy),gray);arrow(d,(x+505,cy),(x+475,cy),gray)
 d.ellipse((cx-7,cy-7,cx+7,cy+7),fill=navy)
 text(d,(x+110,90),label,27,bold=True);text(d,(x+140,390),'Same desired centre',24)
text(d,(45,460),'Top-view schematic: align along the box length so the jaws close across its width.',25,color=gray)
im.save(P/'jaw_orientation.png')

pages=[('1. IK: choosing how the arm should bend','fk_ik.png',[
('Start with the two opposite questions',
'Forward kinematics asks: “If I know every joint angle, where is the gripper?” Inverse kinematics (IK) asks: “If I know where I want the gripper, what joint angles could put it there?” The second problem is harder because several joints affect the same endpoint, and more than one arm posture can sometimes reach the same place.'),
('Example from your table',
'The overhead image gives an estimated box location. Calibration expresses that location relative to the robot base. The grasp planner adds a desired approach height and jaw orientation. IK then searches for shoulder, elbow and wrist angles that put the model’s gripper reference near that target. The gripper opening is handled separately.'),
('The input and output',
'Input: a target position, a useful orientation, the current/reference joint angles and angle bounds. Output: a candidate vector of five arm-joint angles, plus position and orientation errors. The sixth actuator opens the jaws; it is not another independent arm-orientation joint. Internally the model uses radians, while the interface may show degrees.'),
('A simple mathematical description',
'Let q be the five arm angles, and let f(q) be the gripper position calculated by the robot model. Forward kinematics evaluates p = f(q). IK seeks q such that f(q) is close to the requested position p_target. For grasping, we also compare orientation. The diagram is schematic: its link lengths and drawn angles are not calibrated SO-101 measurements.'),
]),('2. The SciPy–MuJoCo search loop','solver_loop.png',[
('What SciPy actually does',
'SciPy’s scipy.optimize.least_squares() is the numerical search engine. It proposes joint angles, evaluates the errors returned by our Python function, and chooses new angles that try to reduce the sum of squared errors. Joint bounds restrict the search. This is iterative calculation, not a lookup of a previously learned grasp.'),
('What MuJoCo actually does',
'For each candidate, ReachSolver.position() writes the angles into data.qpos and calls mujoco.mj_forward(). MuJoCo updates the model’s geometric state. Our code reads the gripper reference position from data.site_xpos and its orientation from data.xmat. It can do this without rendering an image and without moving the real robot.'),
('Our Python connects the two',
'The residual function subtracts the desired position from the calculated position and adds weighted orientation errors. It returns those numbers to SciPy. After SciPy proposes another candidate, MuJoCo evaluates that one. The process repeats until the solver stops; our code then independently checks whether the result is good enough.'),
('Calculation and display are separate',
'The phrase “MuJoCo calculates and displays the resulting poses” covers two jobs. During optimization, MuJoCo primarily calculates poses numerically. The viewer later draws the accepted pose or a planned sequence. The arm does not need to try every candidate physically, and the viewer does not need to show every internal optimizer guess.'),
]),('3. How the grasp objective works in our code','jaw_orientation.png',[
('A box needs more than the correct centre',
'The gripper could reach the correct centre while pointing sideways or closing along the wrong direction. solve_natural_grasp() therefore compares three things: gripper position, a gripper axis aligned with the box’s long axis, and a model axis aligned with the table normal. Together these encourage a useful approach and a pinch across the box width.'),
('The actual residual used by the natural-grasp solver',
'r(q) = [ p(q) − target; 0.03 × (y(q) − sign × object_axis); 0.003 × (z(q) − table_up) ]. Here y and z are columns of the gripper’s model rotation matrix. The weights balance position and axis alignment numerically; they are not physical stiffnesses or probabilities. SciPy minimizes the squared size of this residual.'),
('Why both wrist-roll directions are searched',
'The same long-axis alignment can be represented by two opposite signs. Our solver tries both signs and separately searches roll angles below and above the reference roll angle. This addresses your observation that the wrist sometimes rolled too far one way when the other way was available. Multiple starting guesses improve the chance of finding an acceptable solution; they do not guarantee a global optimum.'),
('What counts as acceptable here',
'The natural-grasp function rejects candidates with model position error above 2 mm, jaw-axis misalignment above 15 degrees, or model-axis tilt above 30 degrees. Accepted candidates are ranked with a motion cost that particularly discourages large wrist-roll changes. These are this function’s model-space criteria; they are not measured physical accuracy guarantees, and later path checks can still reject a candidate.'),
]),('4. From a solved pose to a real grasp',None,[
('Why we cannot request every possible orientation',
'An arbitrary rigid tool pose has three position and three orientation degrees of freedom. Your arm has five joints for moving/orienting the gripper, plus the jaw actuator. It therefore cannot independently satisfy every possible six-dimensional pose. Our solver allows useful orientation tolerances instead of demanding an exact arbitrary rotation at every reachable point.'),
('How the solver knows which way to change an angle',
'The local relationship between small joint changes and fingertip movement is called a Jacobian: Δp ≈ J(q) Δq. Think of it as describing which way the tip moves if each joint turns a little. The position-only ReachSolver.solve() supplies SciPy with a Jacobian from mujoco.mj_jacSite(). The natural-grasp solver does not supply that analytic Jacobian; SciPy estimates the residual derivatives numerically.'),
('An illustrative error calculation',
'Suppose the target is [180, 40, 70] mm in robot coordinates. A candidate pose evaluates to [172, 46, 82] mm. Its position residual is [−8, +6, +12] mm, with a distance error of about 15.6 mm. A later candidate at [179, 40.5, 70.8] mm is about 1.37 mm away. That could pass the 2 mm position criterion, but orientation, limits and the path must still pass. These numbers illustrate the arithmetic; they are not a recorded robot trial. Error need not decrease at every trial evaluation.'),
('IK provides a destination pose, not a complete journey',
'A valid endpoint can still require an unsuitable path. Our planner constructs approach, descent, closure, transfer and release stages; generates timed joint trajectories; and checks sampled geometry. Only then can the preview show the planned motion. The control worker sends the approved sequence through the LeRobot motor interface and checks measured joint feedback.'),
('Why a good preview may miss the physical box',
'The solver can be accurate relative to an imperfect model. Camera/base registration error, joint offsets, holder geometry, motor tracking, compliance or an inaccurate object height can cause a real miss. Your camera-left correction and descent adjustments addressed practical discrepancies. Tightening the IK tolerance alone would not fix a target expressed in the wrong physical location.'),
('In one sentence',
'SciPy asks “which angles should we try next?”, MuJoCo answers “those angles put the model here”, our Python judges the mismatch, and the real controller moves only after a usable path has been prepared.'),
])]
# Additional documentation: proposed design, not an installed runtime cache.
im=Image.new('RGB',(1600,730),'#f4f8fa');d=ImageDraw.Draw(im)
text(d,(45,25),'PROPOSED: REMEMBER SOLUTIONS, REFINE FOR THE CURRENT TARGET',27,bold=True)
boxes=[(50,105,420,165,'Fresh target','Current box location + direction'),(600,105,420,165,'Look in memory','Find nearby compatible poses'),(1130,105,420,165,'Try cached angles','MuJoCo checks actual residual'),(1130,405,420,180,'Refine if needed','SciPy starts from cached angles'),(600,405,420,180,'Check current path','Current arm, plate and load'),(50,405,420,180,'Preview / execution','Save useful accepted solutions')]
for x,y,w,h,title,sub in boxes:
 d.rounded_rectangle((x,y,x+w,y+h),18,fill='white',outline=teal,width=4)
 text(d,(x+18,y+25),title,27,bold=True);text(d,(x+18,y+83),sub,21)
arrow(d,(470,185),(600,185));arrow(d,(1020,185),(1130,185));arrow(d,(1340,270),(1340,405));arrow(d,(1130,490),(1020,490));arrow(d,(600,490),(470,490))
text(d,(65,645),'No useful cache result? Fall back to the existing multi-start IK search.',26,bold=True)
im.save(P/'memory_loop.png')
pages.extend([
('5. Yes: IK can benefit from memory','memory_loop.png',[
('Warm start: begin near a previous answer',
'Instead of solving every target from a generic posture, use a previous solution as the initial guess. SciPy calls this input x0. If the box moves a small distance, the previous grasp angles may already be close to a new solution, so fewer adjustments may be needed. They remain a guess: the new target and orientation are still evaluated. [M2]'),
('A persistent cache: remember more than the last answer',
'Store many solved poses with their joint angles. For a new target, retrieve a few similar entries and try them before the ordinary seeds. MoveIt already provides a cached-IK plugin using this general strategy, with fallback and disk persistence. We could apply the idea directly to our SciPy/MuJoCo code without adopting ROS or MoveIt. [M1]'),
('Example: your box moves 1 cm',
'Imagine the box was grasped at location A, then moves 10 mm right with similar height and orientation. The stored grasp at A becomes a useful seed for target B. MuJoCo evaluates that seed, SciPy adjusts the angles, and the current path is checked. The memory saves search effort; it does not tell the arm to grasp the old location.'),
('Exact reuse is a special case',
'If cached angles already satisfy the freshly measured target, current joint bounds and orientation tolerances, optimization for that endpoint may be skipped. This must be established by evaluating the candidate, not by trusting a rounded lookup key. Path generation, feedback and current scene checks still remain.'),
]),
('6. What to remember, and what to recheck',None,[
('An IK entry should contain more than XYZ',
'A proposed entry stores the achieved model position and gripper axes, the five arm angles, relevant gripper state, target/task description, residual errors and a model/version identifier. It should retain the requested target as well as the achieved pose. Tag evidence separately: solver-valid, path-checked in a particular scene, or physically confirmed. A simulated solution must not acquire a physical-success label automatically.'),
('Keep alternative arm configurations',
'For one grasp target, retain distinct wrist-roll or arm branches when they are useful. Your arm may approach the same box from sitting or from its first plate placement. The previous branch may involve a long roll or a blocked approach from the new start. Rank retrieved candidates using current joint travel and the task constraints, then try another branch when a path fails.'),
('How to find a nearby entry efficiently',
'For a small cache, NumPy distance calculations are simple and may be adequate. For a larger cache, scipy.spatial.cKDTree can retrieve nearby positions quickly; orientation and branch compatibility can then filter and rank that shortlist. Position alone is insufficient: a nearby target rotated 90 degrees can require a different grasp. Position and angular distances need explicit scales rather than mixing metres and radians arbitrarily. [M3]'),
('Symmetry helps, but joint limits still matter',
'A box’s long axis may be treated as equivalent after a 180-degree reversal when the task allows it. That does not make every wrist joint angle physically interchangeable modulo a full turn. Calibrated limits, the reference roll and cable constraints still matter. Likewise, averaging two valid joint vectors can produce an invalid pose; interpolation is at most another candidate to refine and check.'),
('Moving the plate does not erase all kinematic knowledge',
'A robot-base-coordinate pose-to-angle relationship can remain useful when objects move. However, old image coordinates, target estimates, path clearance and destination occupancy become stale. We must obtain a new target and rebuild/recheck the motion against the current plate and any deposited object. The IK cache and a full-trajectory cache therefore have different validity conditions.'),
('Changes to the model or calibration',
'Changing link geometry or the tool reference requires invalidating or revalidating kinematic entries. Changing joint mapping or hardware limits also requires revalidation before physical use. Moving the overhead camera invalidates old image-to-base localization, but need not erase pure model-space IK seeds. Version these layers separately so that useful geometry memory does not preserve stale physical assumptions.'),
]),
('7. A practical design for this project',None,[
('What our implementation already remembers',
'ReachSolver.solve_natural_grasp() already tries the supplied reference pose, a joint-range midpoint and, when available, a position-only IK seed. mirror_grasp_plan.py also uses calibration/reach_reference.json to retarget a previously useful approach. These are useful forms of reuse, but the inspected code has no general persistent nearest-pose IK cache. This chapter proposes one; the controller has not been changed for this document.'),
('Recommended first version',
'Add a small on-demand cache around solve_natural_grasp(). Query it using the freshly computed base-frame target and grasp axes; try several compatible seeds with current bounds; validate candidates using the existing residual tests. Build and check the complete staged path. If cached candidates fail, use the current multi-start search. Save sufficiently distinct successful solutions and their provenance, with a size cap and versioned storage.'),
('Why just adding more seeds may not make it faster',
'The current function evaluates combinations of seeds, jaw-axis signs and roll directions before ranking candidates. Appending cached seeds while still evaluating every original combination can increase work. A real fast path must try cached candidates first and avoid unnecessary later searches once an acceptable complete plan exists. That may trade exhaustive motion-cost comparison for lower latency; fallback preserves alternatives when the first candidate fails.'),
('Should we precompute the whole table at startup?',
'A modest offline sample of table positions, approach heights and box directions could seed a cache without moving the real robot. Each sample still needs model validation. Exhaustive sampling of every position, orientation and arm branch grows quickly and includes unreachable poses. For this prototype, an initially empty cache that grows during use is a simpler starting point; targeted precomputation can later fill frequently used regions.'),
('Persist and maintain the cache',
'NumPy NPZ or SQLite could store the proposed entries; a nearest-neighbor index can be rebuilt in memory after loading. Remove near-duplicates while preserving useful branches, and bound cache size. Solver failures should not be permanently remembered as “unreachable”: failure can depend on a starting guess, an old scene or a particular orientation. Persistent cached IK is still a numerical technique and does not require a neural network.'),
('Keep physical outcome memory separate',
'We can also record whether a real approach or grasp succeeded, using operator confirmation where automatic verification is absent. That may guide which seed to try first, but it should not automatically change geometric calibration, descend farther or tighten the jaws. “The model reached this pose” and “the real object was held successfully” remain different evidence.'),
]),
('8. Expected benefit and how to measure it',None,[
('Faster IK does not always mean much faster planning',
'Total planning time also includes perception, geometry checks, destination fitting and candidate paths. For an illustrative 10-second plan with 2 seconds spent in IK, making IK five times faster gives 8 + 2/5 = 8.4 seconds total: about 1.19 times faster overall. These are explanatory numbers, not measurements from our arm. If IK dominates, the benefit could be much larger; if lookup overhead exceeds saved work, caching can be slower.'),
('Benchmark this setup, rather than promise a multiplier',
'Use identical saved scenes and starting joint states for baseline and cached runs. Measure an empty-cache first pass, repeated targets with a warm cache, nearby moved targets, unseen regions and alternative wrist-roll cases. Record total planning time, IK time, actual residual evaluations, cache hit/fallback rates, position/orientation errors, path acceptance and joint travel. Timing only successful easy cache hits would give a misleading result.'),
('Check difficult and changed conditions',
'Include a rotated box, a moved plate, a first object already deposited, a changed initial arm pose and incompatible model/mapping versions. Confirm that fresh localization and fallback work and that cache entries cannot bypass the existing path checks. Compare physical behavior separately after software evaluation; a faster solver must not be reported as improved physical grasp accuracy without measurements.'),
('Other memory-based options',
'For short successive Cartesian steps, seed each IK query from the previous step’s solution; this often encourages a continuous branch, while still requiring checks near limits or singularities. A reusable trajectory library is a larger extension because it depends on start pose, obstacles and payload. A learned pose-to-angle model is another possible future approach, but it adds training and validation requirements and is unnecessary for a useful first cache.'),
('Recommendation for our arm',
'Begin with a small persistent cache used as a prioritized warm start, retain alternative wrist-roll solutions, and keep fresh scene/path validation. Add selective precomputation only if timing measurements show a benefit. This directly implements your idea of doing more work on new problems and reusing that work when similar problems return.'),
('Primary references and scope',
'[M1] MoveIt Cached IK plugin README: persistent cached seeds, fallback and performance tradeoffs. https://github.com/moveit/moveit/blob/master/moveit_kinematics/cached_ik_kinematics_plugin/README.md\n[M2] SciPy least_squares: initial guess x0 and bounded solving. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html\n[M3] SciPy cKDTree: nearest-neighbor indexing. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html\nConsulted 20 September 2026. The proposed design and examples are specific engineering suggestions, not an implemented or benchmarked cache in this repository.'),
])])

plain=['INVERSE KINEMATICS IN YOUR SO-101 SETUP','Expanded explanation based on simulated_reach.py and the current planning architecture.','']
for title,fig,sections in pages:
 plain.extend([title,'='*len(title),''])
 if fig:plain.extend(['Sketch: '+fig,''])
 for h,b in sections:plain.extend([h,b,''])
(P/'IK_Explained.txt').write_text('\n'.join(plain))
for n,f in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(n,F+f))
styles={
 'title':ParagraphStyle('title',fontName='Bold',fontSize=17,leading=22,textColor=colors.HexColor(navy),spaceAfter=10),
 'h':ParagraphStyle('h',fontName='Bold',fontSize=10.3,leading=14,textColor=colors.HexColor(teal),spaceBefore=9,spaceAfter=4,keepWithNext=True),
 'b':ParagraphStyle('b',fontName='Body',fontSize=9.6,leading=13.5,textColor=colors.HexColor(navy),spaceAfter=4),
}
story=[]
for i,(title,fig,sections) in enumerate(pages):
 if i:story.append(PageBreak())
 story.append(Paragraph(html.escape(title),styles['title']))
 if fig:
  im=Image.open(P/fig);w=505;h=w*im.height/im.width
  story += [PDFImage(str(P/fig),width=w,height=h),Spacer(1,8)]
 for h,b in sections:story += [Paragraph(html.escape(h),styles['h']),Paragraph(html.escape(b).replace('\n','<br/>'),styles['b'])]
def footer(c,doc):
 c.setFont('Body',8);c.setFillColor(colors.HexColor(gray));c.drawString(43,25,'SOARM–MuJoCoMirror | IK explained | 20 September 2026');c.drawRightString(A4[0]-43,25,str(doc.page))
SimpleDocTemplate(str(P/'IK_Explained.pdf'),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=35,bottomMargin=43,title='Inverse Kinematics in Your SO-101 Setup',author='Prepared for Saeed').build(story,onFirstPage=footer,onLaterPages=footer)
print('Created',P/'IK_Explained.pdf')
