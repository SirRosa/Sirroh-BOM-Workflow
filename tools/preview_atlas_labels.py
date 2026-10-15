#!/usr/bin/env python3
"""Render a clearly labeled, NON-DESTRUCTIVE review sheet from atlas_manifest.json.

No edits are made to tile PNGs, input atlases, their SHA or the JSON manifest.
Placement is always determined by (row, column), never by names/order.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCHEMA = 'sirroh.atlas.tiles.v1'


def _font(size: int, bold: bool = False):
    filename = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def _measure(draw, text, font):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def _wrap(draw, text, font, width, limit=3):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text.replace('_', ' _ '))
    words = text.split()
    if not words:
        return ['(brak)']
    lines, line = [], ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if _measure(draw, candidate, font)[0] <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = word
        while line and _measure(draw, line, font)[0] > width:
            n = len(line)
            while n > 1 and _measure(draw, line[:n] + '-', font)[0] > width:
                n -= 1
            lines.append(line[:n] + '-')
            line = line[n:]
    if line:
        lines.append(line)
    if len(lines) > limit:
        lines = lines[:limit]
        last = lines[-1]
        while last and _measure(draw, last + '…', font)[0] > width:
            last = last[:-1]
        lines[-1] = last + '…'
    return lines


def render_preview(manifest_path: Path, output: Path, label_mode: str='name',
                   cell_width: int=184, label_height: int=60):
    manifest_path, output = Path(manifest_path), Path(output)
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA:
        raise ValueError('Unsupported manifest schema')
    cols, rows = int(data['grid']['columns']), int(data['grid']['rows'])
    tw, th = map(int, data['tile_size'])
    if cell_width < tw + 8 or label_height < 30:
        raise ValueError('Cell width / label height too small to preserve legibility')
    tiles = data['tiles']
    if len(tiles) != cols * rows:
        raise ValueError('Incomplete manifest: missing tiles')
    ids = {(t['row'], t['column']) for t in tiles}
    if len(ids) != cols*rows:
        raise ValueError('Duplicate tile coordinate')

    pad, head, foot, sep = 16, 65, 30, 5
    step_x, step_y = cell_width + sep, th + label_height + sep
    w, h = 2*pad+cols*step_x-sep, head+rows*step_y-sep+foot
    im = Image.new('RGBA', (w, h), (28, 31, 33, 255))
    d = ImageDraw.Draw(im)
    titlefont, font, small = _font(23, True), _font(13), _font(11)
    d.text((pad, 12), 'SIRROH BOM  ·  PODGLĄD PREFABÓW', font=titlefont, fill=(242,237,221))
    d.text((pad, 44), f'{cols} × {rows}  |  {len(tiles)} kafelków  |  {label_mode}  |  tylko podgląd — nie używać jako atlasu gry', font=small, fill=(160,174,179))
    root = manifest_path.parent.resolve()
    for t in tiles:
        r,c=int(t['row'])-1,int(t['column'])-1
        if not (0<=r<rows and 0<=c<cols):
            raise ValueError('Invalid tile coordinates')
        x,y=pad+c*step_x, head+r*step_y
        base=(42,46,47,255) if not t.get('empty') else (55,52,48,255)
        d.rectangle((x,y,x+cell_width-1,y+th+label_height-1),fill=base,outline=(106,115,111),width=1)
        tilepath=(root/t['file']).resolve()
        if not tilepath.is_relative_to(root):
            raise ValueError('Unsafe tile path')
        with Image.open(tilepath) as src:
            if src.size!=(tw,th) or src.mode!='RGBA':
                raise ValueError(f'Invalid tile: {tilepath}')
            sprite=src.copy()
        ix=x+(cell_width-tw)//2
        im.alpha_composite(sprite,(ix,y))
        d.rectangle((x+5,y+5,x+65,y+24),fill=(15,17,18,240))
        d.text((x+10,y+7),t['id'],font=small,fill=(225,229,216))
        d.line((x+1,y+th,x+cell_width-2,y+th),fill=(102,112,112),width=1)
        name=str(t.get('name') or t['id'])
        prefab=str(t.get('prefab') or '')
        if label_mode=='prefab':
            name=prefab or name
        elif label_mode=='both' and prefab and prefab!=name:
            name+=' / '+prefab
        elif label_mode not in ('name','prefab','both'):
            raise ValueError('Unknown label mode')
        lines=_wrap(d,name,font,cell_width-13,3)
        lineheight=17
        top=y+th+max(3,(label_height-len(lines)*lineheight)//2)
        color=(238,238,223) if prefab else (236,186,136)
        if t.get('empty'):
            color=(161,161,156)
        for line in lines:
            ww,_=_measure(d,line,font)
            d.text((x+(cell_width-ww)//2,top),line,font=font,fill=color)
            top+=lineheight
    d.text((pad,h-23),'Białe podpisy: identyfikatory prefabów    ·    bursztynowe: custom_*',font=small,fill=(197,197,185))
    output.parent.mkdir(parents=True,exist_ok=True)
    im.convert('RGB').save(output,'PNG',optimize=True)
    print(f'PREVIEW OK: {output} | {w}x{h} | {cols*rows} slots')
    return output


def main():
    p=argparse.ArgumentParser(description='Review-only atlas with tile IDs and labels')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label-mode',choices=['name','prefab','both'],default='name')
    p.add_argument('--cell-width',type=int,default=184)
    p.add_argument('--label-height',type=int,default=60)
    a=p.parse_args()
    render_preview(a.manifest,a.output,a.label_mode,a.cell_width,a.label_height)

if __name__=='__main__':main()
