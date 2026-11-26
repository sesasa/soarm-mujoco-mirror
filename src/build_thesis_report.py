"""Build the thesis-style report from Markdown and copied source evidence.
Run from repository root: .venv/bin/python src/build_thesis_report.py
No robot, camera or control modules are imported.
"""
from pathlib import Path
import sys, json, hashlib, platform, importlib.metadata, shutil, re, html, math
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE=Path(__file__).resolve().parents[1]/'reports/soarm_thesis'
ROOT=HERE.parents[1]
ASSETS=HERE/'assets'; ASSETS.mkdir(exist_ok=True)
FONT='/usr/share/fonts/truetype/dejavu/'
for name,file in [('Body','DejaVuSerif.ttf'),('BodyBold','DejaVuSerif-Bold.ttf'),('Sans','DejaVuSans.ttf'),('SansBold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
 pdfmetrics.registerFont(TTFont(name,FONT+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansBold',italic='Sans',boldItalic='SansBold')
NAVY='#153247'; TEAL='#087F8C'; GRAY='#526371'

def font(size,bold=False):return ImageFont.truetype(FONT+('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),size)
def diagram(name,boxes,arrows,size=(1700,850)):
 im=PILImage.new('RGB',size,'#f5f8fa');d=ImageDraw.Draw(im)
 for x,y,w,h,title,subtitle in boxes:
  d.rounded_rectangle((x,y,x+w,y+h),20,fill='white',outline=TEAL,width=4)
  d.multiline_text((x+w/2,y+35),title,font=font(30,True),fill=NAVY,anchor='mt',align='center',spacing=8)
  d.multiline_text((x+w/2,y+95),subtitle,font=font(24),fill=GRAY,anchor='mt',align='center',spacing=8)
 for points in arrows:
  d.line(points,fill=TEAL,width=5)
  a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);s=18
  d.polygon([b,(b[0]-s*math.cos(angle-.5),b[1]-s*math.sin(angle-.5)),(b[0]-s*math.cos(angle+.5),b[1]-s*math.sin(angle+.5))],fill=TEAL)
 im.save(ASSETS/name)

diagram('architecture.png',[
 (40,45,440,175,'Fixed overhead camera','Calibration + current image'),
 (625,45,440,175,'Scene perception','Object footprints + plate'),
 (1210,45,440,175,'MuJoCo + SciPy','Kinematics + path preview'),
 (1210,380,440,185,'Control worker','LeRobot motor interface'),
 (625,380,440,185,'Physical SO-101','Joint feedback + movement'),
 (40,380,440,185,'Operator / wrist view','Inspect, execute, hold'),
 ], [[(480,130),(625,130)],[(1065,130),(1210,130)],[(1430,220),(1430,380)],[(1210,470),(1065,470)],[(480,470),(565,470),(565,300),(1430,300)],[(845,565),(845,695),(1560,695),(1560,565)]],size=(1700,760))
# Add feedback caption in reserved lower band.
p=ASSETS/'architecture.png';im=PILImage.open(p);d=ImageDraw.Draw(im);d.text((985,710),'Measured joint feedback',font=font(25),fill=GRAY);im.save(p)
diagram('batch_flow.png',[
 (35,40,495,185,'Fresh initial scene','Packet + box + destination'),
 (610,40,495,185,'Complete preview','Both objects / return at end'),
 (1185,40,495,185,'Execute first object','Grasp → place → open'),
 (1185,370,495,185,'Visibility-aware exit','Reveal plate and next object'),
 (610,370,495,185,'Fresh second plan','New image + current joints'),
 (35,370,495,185,'Execute second object','Place, then return to sitting'),
 ], [[(530,130),(610,130)],[(1105,130),(1185,130)],[(1430,225),(1430,370)],[(1185,465),(1105,465)],[(610,465),(530,465)]],size=(1720,620))

im=PILImage.new('RGB',(1700,650),'white');d=ImageDraw.Draw(im)
for x0,title,fun,maxv in [(130,'Position fraction',lambda u:10*u**3-15*u**4+6*u**5,1),(930,'Normalized velocity',lambda u:30*u**2-60*u**3+30*u**4,2)]:
 w,h=610,390;y0=510
 d.text((x0,35),title,font=font(32,True),fill=NAVY)
 for j in range(5):
  y=y0-h*j/4;d.line((x0,y,x0+w,y),fill='#dbe5e9',width=2);d.text((x0-65,y-14),f'{maxv*j/4:g}',font=font(22),fill=GRAY)
 d.line((x0,110,x0,y0,x0+w,y0),fill=NAVY,width=3)
 pts=[(x0+w*i/200,y0-h*fun(i/200)/maxv) for i in range(201)];d.line(pts,fill=TEAL,width=6)
 for j in range(5): d.text((x0+w*j/4-15,y0+12),f'{j/4:g}',font=font(22),fill=GRAY)
 d.text((x0+90,580),'Normalized segment time u',font=font(26),fill=GRAY)
im.save(ASSETS/'trajectory.png')

images={
 'holder_alignment.jpg':'captures/calibrated_preview/holder_photo_shape.jpg',
 'two_objects.jpg':'captures/mirror_grasp/1789856005604745428/latest.jpg',
 'contact_simulation.jpg':'captures/contact_preview/latest.jpg',
}
for dest,src in images.items():shutil.copy2(ROOT/src,ASSETS/dest)
# Frozen environment and evidence inventory, without importing hardware modules.
versions={}
for dist in ['mujoco','opencv-python','numpy','scipy','pillow','reportlab','glfw','PyOpenGL']:
 try:versions[dist]=importlib.metadata.version(dist)
 except importlib.metadata.PackageNotFoundError:versions[dist]='not found'
paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'config').glob('*.json'))
paths += [ROOT/p for p in ['requirements.txt','assets/so101/README.md','assets/so101/LICENSE','calibration/cameras/overhead_640x480.json','calibration/cameras/wrist_640x480.json','calibration/robot_camera/overhead_from_base_provisional.json','calibration/planning_performance.json','captures/contact_preview/lift_result.json','captures/contact_preview/cycle_result.json']]
manifest={'date':'2026-09-20','python':sys.version,'platform':platform.platform(),'package_versions':versions,'image_sources':images,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},'configuration':{p.name:json.loads(p.read_text()) for p in (ROOT/'config').glob('*.json')},'report_test_output':(HERE/'test_results.txt').read_text()}
(HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

W,H=A4;M=51;CONTENT=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10,leading=13.5,spaceAfter=7,textColor=colors.HexColor(NAVY)),
 'h1':ParagraphStyle('h1',fontName='SansBold',fontSize=19,leading=24,spaceAfter=16,textColor=colors.HexColor(NAVY)),
 'h2':ParagraphStyle('h2',fontName='SansBold',fontSize=11.7,leading=15,spaceBefore=10,spaceAfter=6,keepWithNext=True,textColor=colors.HexColor(TEAL)),
 'cap':ParagraphStyle('cap',fontName='Sans',fontSize=8.4,leading=12,spaceBefore=6,spaceAfter=14,textColor=colors.HexColor(GRAY)),
 'cell':ParagraphStyle('cell',fontName='Sans',fontSize=8,leading=10.5,textColor=colors.HexColor(NAVY)),
 'headcell':ParagraphStyle('headcell',fontName='SansBold',fontSize=8,leading=10.5,textColor=colors.white),
}
styles['compactbody']=ParagraphStyle('compactbody',parent=styles['body'],fontSize=9.5,leading=12.1,spaceAfter=5)
styles['compacth2']=ParagraphStyle('compacth2',parent=styles['h2'],spaceBefore=8,spaceAfter=5)
styles['compactcap']=ParagraphStyle('compactcap',parent=styles['cap'],leading=10.5,spaceAfter=9)
def inline(s):
 s=html.escape(s)
 s=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8.4">\1</font>',s)
 # URL break opportunities preserve readable references and clickable links.
 s=re.sub(r'(https://[^\s<]+)',lambda m:'<link href="'+m.group(1)+'" color="#087F8C">'+m.group(1)+'</link>',s)
 return s

def furniture(c,doc):
 c.saveState()
 if doc.page>1:
  c.setStrokeColor(colors.HexColor('#c9d7df'));c.setLineWidth(.5);c.line(M,H-40,W-M,H-40)
  c.setFont('Sans',8);c.setFillColor(colors.HexColor(GRAY));c.drawString(M,H-31,'SOARM–MuJoCoMirror  |  Engineering development report')
  c.drawString(M,29,'20 September 2026  •  Source and evidence snapshot')
  c.drawRightString(W-M,29,str(doc.page))
 c.restoreState()
class Doc(BaseDocTemplate):
 def __init__(self,path):
  super().__init__(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=60,bottomMargin=49,title='SOARM–MuJoCoMirror: A Camera-Informed Desktop Manipulation System',author='Prepared for Saeed',subject='Thesis-style project report: calibration, vision, simulation and real-arm manipulation')
  self.addPageTemplates(PageTemplate(id='normal',frames=[Frame(M,49,CONTENT,H-109,id='main',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=furniture))
 def afterFlowable(self,f):
  if isinstance(f,Paragraph) and getattr(f,'toc_entry',None):
   title,key=f.toc_entry;self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,0);self.notify('TOCEntry',(0,title,self.page,key))

story=[]
cover=ParagraphStyle('cover',fontName='SansBold',fontSize=31,leading=39,textColor=colors.HexColor(NAVY))
sub=ParagraphStyle('sub',fontName='Sans',fontSize=16,leading=24,textColor=colors.HexColor(TEAL))
small=ParagraphStyle('small',fontName='Sans',fontSize=10,leading=16,textColor=colors.HexColor(GRAY))
story += [Spacer(1,45),Paragraph('SOARM–<br/>MuJoCoMirror',cover),Spacer(1,15),Paragraph('A camera-informed desktop<br/>manipulation system',sub),Spacer(1,24),Paragraph('From joint mirroring and camera calibration<br/>to simulation-mediated pick and place',small),Spacer(1,35)]
p=ASSETS/'two_objects.jpg';im=PILImage.open(p);story.append(Image(str(p),width=CONTENT,height=CONTENT*im.height/im.width));story += [Spacer(1,28),Paragraph('THESIS-STYLE ENGINEERING REPORT',small),Spacer(1,9),Paragraph('Prepared for Saeed<br/>20 September 2026',small),Spacer(1,18),Paragraph('Motivations • Methods • Python implementation<br/>Experimental evidence • AI/NN role • Limitations',small),PageBreak()]
story.append(Paragraph('Contents',styles['h1']))
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='Sans',fontSize=9.5,leading=12.5,spaceBefore=5,rightIndent=20)];story.extend([toc,PageBreak()])
lines=(HERE/'report.md').read_text().splitlines();i=0;chapter=0;compact=False
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('# '):
  if chapter:story.append(PageBreak())
  title=line[2:];compact=bool(re.match(r'^(8|10|12|13)\.',title));p=Paragraph(inline(title),styles['h1']);p.toc_entry=(title,'chapter'+str(chapter));story.append(p);chapter+=1;i+=1;continue
 if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['compacth2' if compact else 'h2']));i+=1;continue
 if line.startswith('!['):
  match=re.match(r'!\[(.*?)\]\((.*?)\)',line);caption,filename=match.groups();p=HERE/filename;im=PILImage.open(p);width=CONTENT;height=width*im.height/im.width
  maxheight=170 if compact else 210
  if height>maxheight:width*=maxheight/height;height=maxheight
  story.append(KeepTogether([Spacer(1,8),Image(str(p),width=width,height=height),Paragraph(inline(caption),styles['compactcap' if compact else 'cap'])]));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.match(r'^[-: ]+$',x) for x in cells):rows.append(cells)
   i+=1
  n=len(rows[0]);ratios={2:[.43,.57],3:[.28,.35,.37],4:[.25,.16,.23,.36]}[n]
  data=[[Paragraph(inline(x),styles['headcell' if ri==0 else 'cell']) for x in row] for ri,row in enumerate(rows)]
  t=Table(data,colWidths=[CONTENT*r for r in ratios],repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#c9d7df'))]))
  story.extend([t,Spacer(1,13)]);continue
 para=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
  para.append(lines[i].strip());i+=1
 story.append(Paragraph(inline(' '.join(para)),styles['compactbody' if compact else 'body']))
Doc(HERE/'SOARM_MuJoCoMirror_Thesis_Report.pdf').multiBuild(story)
print('Built',HERE/'SOARM_MuJoCoMirror_Thesis_Report.pdf')
