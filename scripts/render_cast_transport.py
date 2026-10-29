#!/usr/bin/env python3
"""Render a reproducible CAST distribution-fitting animation (NumPy + Pillow).

Synthetic fixed critic, two-dimensional action space, one fixed observation.
The residual actor is optimized against frozen targets with gradient descent.
Rendered frames interpolate actual saved optimization steps. Not robot results.
Requires numpy, pillow, and ffmpeg (or imageio-ffmpeg). See --help.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/animations'
W, H, FPS = 1200, 860, 24
DURATION = 27
BG = '#0b2029'
PANEL = '#102b35'
TEXT = '#f0f6f3'
MUTED = '#9bb6b9'
TEAL = '#69ebbd'
BLUE = '#79a9f7'
GOLD = '#ffd583'
LINE = '#25444e'
FONTS = Path('/usr/share/fonts/dejavu-sans-fonts')
PARAMS = dict(seed=2027, candidates=192, rounds=4, fit_steps=12,
              eta_q=1.1, delta_q=1., eta_0=.1, eta_anchor=.65,
              rho=.32, delta_total=.25, critic_sigma=.85)


def clip(v, cap):
    return v * np.minimum(1, cap / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))


def critic(a):
    q = np.ones(len(a)); g = np.zeros_like(a)
    for center in ((.65, -.65), (.65, .65)):
        d = a - center
        e = np.exp(-np.sum(d*d, axis=1) / (2*PARAMS['critic_sigma']**2))
        q += 1.3*e
        g -= 1.3*e[:, None]*d / PARAMS['critic_sigma']**2
    return q, g


def simulate():
    rng = np.random.default_rng(PARAMS['seed'])
    z = rng.normal(size=(PARAMS['candidates'], 3))
    base = np.column_stack((-.95+.28*z[:, 0], .68*np.tanh(3*z[:, 2])+.2*z[:, 1]))
    features = np.column_stack((np.ones(len(z)), base, base[:, 0]*base[:, 1], base[:, 1]**2))
    theta = np.zeros((features.shape[1], 2))
    lr = .7 / np.linalg.eigvalsh(features.T @ features / len(z)).max()
    records = []
    for k in range(PARAMS['rounds']):
        r = features @ theta
        current = base + r
        q, grad = critic(current)
        dq = PARAMS['eta_q']*clip(grad / np.mean(np.abs(q)), PARAMS['delta_q'])
        rms = np.linalg.norm(r, axis=1, keepdims=True) / np.sqrt(2)
        dr = -PARAMS['eta_0']*r - PARAMS['eta_anchor']*np.maximum(1-PARAMS['rho']/np.maximum(rms, 1e-12), 0)*r
        delta = clip(dq+dr, PARAMS['delta_total'])
        target = (r+delta).copy()  # a fixed NumPy array during all fitting steps
        history, losses = [], []
        for j in range(PARAMS['fit_steps']+1):
            error = features @ theta - target
            history.append((base + features @ theta).copy())
            losses.append(float(np.mean(error**2)))
            if j < PARAMS['fit_steps']:
                theta -= lr * (features.T @ error) / len(z)
        assert max(np.linalg.norm(delta, axis=1)) <= PARAMS['delta_total'] + 1e-10
        assert all(a >= b-1e-12 for a, b in zip(losses, losses[1:]))
        records.append(dict(current=current, proposal=current+dq, restored=current+dq+dr,
                            target=base+target, fitted=np.array(history), losses=losses,
                            q_before=float(q.mean()), q_after=float(critic(history[-1])[0].mean()),
                            max_target_step=float(np.linalg.norm(delta, axis=1).max())))
    trace = dict(description=__doc__, parameters=PARAMS, actor_learning_rate=float(lr),
                 base='a0(z) = (-0.95 + 0.28*z1, 0.68*tanh(3*z3) + 0.20*z2), z ~ N(0,I3)',
                 residual='r_theta(z) = [1, a0_x, a0_y, a0_x*a0_y, a0_y^2] @ theta',
                 critic='1 + sum_c 1.3*exp(-||a-c||^2/(2*0.85^2)); c=(0.65,+/-0.65)',
                 display='Gaussian-smoothed empirical densities. Colors and opacity are for visibility. Frames interpolate saved SGD steps; the last fade resets the illustration.',
                 base_actions=base.tolist(),
                 rounds=[{k:(v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in records])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'cast_distribution_trace.json').write_text(json.dumps(trace, separators=(',', ':'))+'\n')
    return base, records


@lru_cache(None)
def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = FONTS / name
    if not path.exists():
        path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size)


def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))


def blend(c, alpha, bg=BG):
    return tuple(round(a*alpha+b*(1-alpha)) for a,b in zip(rgb(c),rgb(bg)))


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x*x*(3-2*x)


def mix(a,b,t):
    return a*(1-t)+b*t


# Fixed axes: every frame uses the same action coordinates.
LEFT, TOP, RIGHT, BOTTOM = 65, 185, 835, 542
XLO, XHI, YLO, YHI = -1.9, 1.65, -1.6, 1.6


def screen(a):
    a = np.asarray(a)
    return np.stack((LEFT+(a[..., 0]-XLO)/(XHI-XLO)*(RIGHT-LEFT),
                     BOTTOM-(a[..., 1]-YLO)/(YHI-YLO)*(BOTTOM-TOP)), axis=-1)


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W,H), BG)
        self.d = ImageDraw.Draw(self.im)
    def text(self,x,y,s,size=22,color=TEXT,bold=False,anchor=None):
        self.d.text((x,y),s,font=font(size,bold),fill=color,anchor=anchor)
    def line(self,pts,color=LINE,width=2):
        self.d.line([tuple(p) for p in pts], fill=color, width=width)
    def dashed(self,pts,color=BLUE,width=2,dash=7):
        for a,b in zip(pts,pts[1:]):
            a,b=np.asarray(a),np.asarray(b);length=np.linalg.norm(b-a)
            if length < 1: continue
            for i in np.arange(0,length,2*dash):
                self.line([mix(a,b,i/length),mix(a,b,min(i+dash,length)/length)],color,width)
    def arrow(self,a,b,color=GOLD,alpha=1,width=2):
        a,b=np.asarray(a),np.asarray(b);norm=np.linalg.norm(b-a)
        if norm < 2:return
        color=blend(color,alpha);self.line([a,b],color,width)
        u=(b-a)/norm;v=np.array([-u[1],u[0]]);s=min(8,norm*.35)
        self.d.polygon([tuple(b),tuple(b-s*u+s*.45*v),tuple(b-s*u-s*.45*v)],fill=color)
    def density(self,a,color=TEAL,alpha=.6,contour=False):
        hist,_,_=np.histogram2d(a[:,0],a[:,1],bins=(210,110),range=((XLO,XHI),(YLO,YHI)))
        kernel=np.exp(-.5*(np.arange(-14,15)/4.5)**2);kernel/=kernel.sum()
        field=np.apply_along_axis(lambda row: np.convolve(row,kernel,mode='same'),0,hist.T[::-1])
        field=np.apply_along_axis(lambda row: np.convolve(row,kernel,mode='same'),1,field)
        field/=max(field.max(),1e-12)
        opacity=field*.32
        for level in (.2,.45,.72):
            opacity=np.maximum(opacity,np.maximum(0,1-np.abs(field-level)/.055)*(.6 if contour else .28))
        if contour:opacity=np.where(field>.1,opacity,0)
        rgba=np.zeros((*field.shape,4),dtype=np.uint8);rgba[:,:,:3]=rgb(color)
        rgba[:,:,3]=np.uint8(np.clip(opacity*alpha*255,0,255))
        layer=Image.fromarray(rgba).resize((RIGHT-LEFT,BOTTOM-TOP),Image.BICUBIC)
        self.im.paste(layer,(LEFT,TOP),layer)
    def particles(self,a,color=TEAL,alpha=1,radius=3,ring=False,subset=None):
        for x,y in screen(a if subset is None else a[subset]):
            if not LEFT <= x <= RIGHT or not TOP <= y <= BOTTOM:continue
            box=(x-radius,y-radius,x+radius,y+radius)
            c=blend(color,alpha)
            if ring:self.d.ellipse(box,outline=c,width=2)
            else:self.d.ellipse(box,fill=c)
    def curve(self,a,color,alpha=1,dashed=False,fill=False):
        xs=np.linspace(XLO,XHI,400)
        bandwidth=.11
        ys=np.exp(-.5*((xs[:,None]-a[:,0])/bandwidth)**2).mean(axis=1)/(bandwidth*np.sqrt(2*np.pi))
        pts=np.column_stack((LEFT+(xs-XLO)/(XHI-XLO)*(RIGHT-LEFT),704-ys*45))
        c=blend(color,alpha)
        if fill:self.d.polygon([tuple(x) for x in pts]+[(RIGHT,704),(LEFT,704)],fill=blend(color,alpha*.13))
        if dashed:
            for i in range(0,len(pts)-1,8):self.line(pts[i:i+5],c,2)
        else:self.line(pts,c,3)


def draw(t,base,records):
    c=Canvas()
    # The reset is a fade, never a reverse optimization trajectory.
    k=min(int(max(0,t-1)/6),len(records)-1)
    u=np.clip(t-1-k*6,0,6)
    stage=0 if u<1 else 1 if u<2.3 else 2 if u<3.7 else 3
    rec=records[k]
    fit_progress=ease((u-3.7)/2.1)
    jf=fit_progress*PARAMS['fit_steps'];j=min(int(jf),PARAMS['fit_steps']-1)
    actor=mix(rec['fitted'][j],rec['fitted'][j+1],jf-j) if stage==3 else rec['current']
    loss=float(np.mean((actor-rec['target'])**2))
    titles=['Every action keeps its own reference.','The critic proposes an improvement.',
            'Restore toward the base. Limit the step.','Fit the policy to the fixed targets.']
    c.text(40,26,'CAST  /  LEARNING IN ACTION',18,TEAL,True)
    c.text(1160,27,f'ROUND  {k+1:02d} / {len(records):02d}',18,MUTED,True,'ra')
    c.text(40,67,titles[stage],32,TEXT,True)
    c.d.rounded_rectangle((36,132,862,736),radius=18,fill=PANEL)
    c.d.rounded_rectangle((882,132,1164,736),radius=18,fill='#122f38')
    c.text(60,149,'ACTION DISTRIBUTION',16,MUTED,True)
    c.text(838,150,'higher predicted value →',15,GOLD,anchor='ra')
    for x in np.linspace(XLO,XHI,8):
        p=screen([x,YLO]);c.line([(p[0],TOP),(p[0],BOTTOM)],'#1a3943',1)
    for y in np.linspace(YLO,YHI,6):
        p=screen([XLO,y]);c.line([(LEFT,p[1]),(RIGHT,p[1])],'#1a3943',1)
    # Critic preference is explicitly predicted value, not a ground-truth reward.
    for center in ((.65,-.65),(.65,.65)):
        x,y=screen(center)
        for r in (14,28,45):
            c.d.ellipse((x-r,y-r*.55,x+r,y+r*.55),outline=blend(GOLD,.18),width=1)
        c.d.ellipse((x-3,y-3,x+3,y+3),fill=blend(GOLD,.5))
    c.density(base,BLUE,.75,True)
    c.particles(base,BLUE,.3,2)
    c.density(actor,TEAL,.85)
    selected=np.arange(6,len(base),13)
    if stage==0:
        for b,a in zip(screen(base[selected]),screen(actor[selected])):
            c.dashed([b,a],blend(BLUE,.55),1,4)
    elif stage==1:
        p=ease((u-1)/1.1)
        proposal=mix(rec['current'],rec['proposal'],p)
        for a,b in zip(screen(rec['current'][selected]),screen(proposal[selected])):
            c.arrow(a,b,GOLD,.8)
        c.particles(proposal,GOLD,.7,3,True,selected)
        c.curve(proposal,GOLD,.65,True)
    elif stage==2:
        p=ease((u-2.3)/.65)
        restored=mix(rec['proposal'],rec['restored'],p)
        target=mix(restored,rec['target'],ease((u-2.95)/.65))
        for a,b,r in zip(screen(rec['current'][selected]),screen(rec['proposal'][selected]),screen(target[selected])):
            c.arrow(a,b,GOLD,.25)
            c.arrow(b,r,BLUE,.85)
            c.arrow(a,r,TEAL,.9)
        c.particles(target,TEXT,.9,5,True,selected)
        c.curve(target,TEAL,.85,True)
    else:
        c.density(rec['target'],TEAL,.25,True)
        c.particles(rec['target'],TEXT,.6,4,True,selected)
        for a,b in zip(screen(actor[selected]),screen(rec['target'][selected])):
            c.line([a,b],blend(TEXT,.4),1)
        c.curve(rec['target'],TEXT,.6,True)
    c.particles(actor,TEAL,.8,2)
    c.particles(actor,TEAL,1,3,False,selected)
    # One highlighted same-noise sample makes the persistent reference visible.
    hi=84;b,a=screen(base[hi]),screen(actor[hi])
    c.dashed([b,a],blend(BLUE,.8),2,5)
    c.d.ellipse((b[0]-6,b[1]-6,b[0]+6,b[1]+6),outline=BLUE,width=2)
    c.d.ellipse((a[0]-6,a[1]-6,a[0]+6,a[1]+6),fill=TEAL,outline=TEXT,width=2)
    c.text(60,565,'The base stays fixed. The fitted distribution changes.',16,MUTED)
    c.line([(60,598),(837,598)],LINE,1)
    c.text(60,611,'DENSITY ALONG ACTION AXIS 1',14,MUTED,True)
    c.curve(actor,TEAL,1,False,True)
    c.curve(base,BLUE,.9,True)
    if stage==3:c.curve(rec['target'],TEXT,.7,True)
    c.line([(LEFT,705),(RIGHT,705)],LINE,1)
    for x in (-1.5,-.5,.5,1.5):
        xx=screen([x,0])[0];c.text(xx,710,f'{x:g}',13,MUTED,anchor='ma')
    labels=['PAIR','PROPOSE','CONSTRAIN','FIT']
    desc=[['Match each action','to its frozen base.'],['Follow a normalized,','capped critic gradient.'],['Add restoration,','then cap the sum.'],['Keep targets fixed.','Optimize the residual.']]
    for i,label in enumerate(labels):
        y=154+i*91;active=i==stage
        if active:c.d.rounded_rectangle((898,y-4,1148,y+77),radius=10,fill='#204d50')
        c.text(912,y+4,f'0{i+1}',17,TEAL if active else MUTED,True)
        c.text(950,y+3,label,18,TEXT if active else MUTED,True)
        c.text(912,y+30,desc[i][0],15,TEXT if active else MUTED)
        c.text(912,y+50,desc[i][1],15,TEXT if active else MUTED)
    c.line([(908,536),(1140,536)],LINE,1)
    if stage==3:
        c.text(910,553,'TARGET FIT ERROR',14,MUTED,True)
        ratio=loss/rec['losses'][0]
        c.text(910,578,f'{loss:.4f}',31,TEAL,True)
        c.text(910,622,f'SGD step {int(jf):02d} / 12',15,MUTED)
        c.d.rounded_rectangle((910,652,1136,662),radius=5,fill='#26434c')
        c.d.rounded_rectangle((910,652,910+max(8,min(1,ratio)*226),662),radius=5,fill=TEAL)
        c.text(910,681,'Targets refresh each round.',13,MUTED)
    else:
        c.text(910,553,'TARGET STEP CAP',14,MUTED,True)
        c.text(910,578,'0.25',31,TEAL,True)
        c.text(910,622,'Normalized action units',14,MUTED)
        c.text(910,656,'A limit on each target step.',13,MUTED)
    # Shared legend and total timeline.
    for x,label,color in ((44,'Frozen base',BLUE),(270,'Current policy',TEAL),(532,'Fixed targets',TEXT),(776,'Critic proposal',GOLD)):
        c.d.ellipse((x,764,x+9,773),fill=color)
        c.text(x+18,755,label,17,MUTED)
    c.line([(40,805),(1160,805)],LINE,3)
    c.line([(40,805),(40+1120*min(t/26,1),805)],TEAL,3)
    c.text(40,824,'Illustrative toy actor · actual target regression · smoothed densities',14,MUTED)
    c.text(1160,824,'PAIR → PROPOSE → CONSTRAIN → FIT',13,TEAL,True,'ra')
    if t>26.3:
        c.im=Image.blend(c.im,Image.new('RGB',(W,H),BG),ease((t-26.3)/.7))
    return c.im


def find_ffmpeg():
    if os.environ.get('FFMPEG'):return os.environ['FFMPEG']
    if shutil.which('ffmpeg'):return shutil.which('ffmpeg')
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit('Install imageio-ffmpeg or set FFMPEG to the ffmpeg executable.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true',help='Render stage posters only.')
    args=parser.parse_args();base,records=simulate()
    for name,t in [('poster',17.9),('pair',13.5),('propose',14.8),('constrain',16.3),('fit',18.4)]:
        draw(t,base,records).save(OUT/f'cast_distribution_{name}.png')
    print('Toy checked: every target respects the step cap; fitting error decreases each SGD step.',flush=True)
    if args.preview:return
    ffmpeg=find_ffmpeg();mp4=OUT/'cast_distribution.mp4'
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(mp4)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(DURATION*FPS):
        proc.stdin.write(draw(i/FPS,base,records).tobytes())
        if i%(FPS*6)==0:print(f'Rendered {i//FPS}s / {DURATION}s',flush=True)
    proc.stdin.close()
    if proc.wait()!=0:raise SystemExit('Video encoding failed')
    # A single global palette avoids color flicker. GIF uses 15 fps / 840 px.
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(mp4),'-filter_complex','fps=15,scale=840:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4','-loop','0',str(OUT/'cast_distribution.gif')],check=True)
    print('Saved MP4, looping GIF, poster, and numerical trace.',flush=True)


if __name__=='__main__':main()
