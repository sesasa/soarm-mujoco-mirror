"""Make exact-size A4 vector checkerboards; no printer is invoked."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4,landscape

ROOT=Path(__file__).resolve().parents[1]

def draw(c,square,label):
    w,h=landscape(A4);cols,rows=10,7
    x=(w-cols*square*mm)/2;y=20*mm if square==25 else 65*mm
    c.setTitle('SOARM camera calibration boards')
    c.setFont('Helvetica',10)
    c.drawCentredString(w/2,201*mm,f'{label} | {square} mm squares | 9 x 6 inner corners | Print at 100%, Actual size')
    for row in range(rows):
        for col in range(cols):
            if (row+col)%2==0:
                c.rect(x+col*square*mm,y+row*square*mm,square*mm,square*mm,stroke=0,fill=1)
    # Two orthogonal scale checks, outside the board white margin.
    bx=25*mm;by=10*mm
    c.setLineWidth(.5);c.line(bx,by,bx+100*mm,by)
    for i in range(11):c.line(bx+i*10*mm,by-1.5*mm,bx+i*10*mm,by+1.5*mm)
    c.setFont('Helvetica',8);c.drawString(bx+105*mm,by-1*mm,'This line must measure exactly 100 mm (10 cm).')
    c.line(10*mm,50*mm,10*mm,150*mm)
    c.line(8.5*mm,50*mm,11.5*mm,50*mm);c.line(8.5*mm,150*mm,11.5*mm,150*mm)
    c.saveState();c.translate(7*mm,70*mm);c.rotate(90);c.drawString(0,0,'100 mm vertical check');c.restoreState()
    if square==10:
        c.setFont('Helvetica',11);c.drawCentredString(w/2,45*mm,'Wrist board: keep the full pattern and a white border visible.')
        c.drawCentredString(w/2,38*mm,'Mount flat on stiff card; do not bend, laminate with gloss, or resize.')
    c.showPage()

def main():
    out=ROOT/'calibration/print';out.mkdir(parents=True,exist_ok=True)
    specs=[(25,'LARGE - overhead camera'),(10,'SMALL - wrist camera')]
    combined=canvas.Canvas(str(out/'camera_calibration_A4.pdf'),pagesize=landscape(A4))
    for square,label in specs:
        draw(combined,square,label)
        single=canvas.Canvas(str(out/f'checkerboard_{square}mm_A4.pdf'),pagesize=landscape(A4));draw(single,square,label);single.save()
    combined.save()
    (ROOT/'config/calibration_boards.json').write_text(json.dumps({'inner_corners':[9,6],'boards':{'large':{'square_m':.025,'pdf':'calibration/print/checkerboard_25mm_A4.pdf'},'small':{'square_m':.010,'pdf':'calibration/print/checkerboard_10mm_A4.pdf'}}},indent=2)+'\n')
    print(out/'camera_calibration_A4.pdf')
if __name__=='__main__':main()
