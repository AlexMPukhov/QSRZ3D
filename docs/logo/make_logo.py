"""QUARZ logo: writes the SVGs of docs/logo with the lettering converted to outlines.
usage: FONTS=<dir with the two fonts> python3 make_logo.py <out dir>
PNGs: rendered from the SVGs with a headless browser."""
import os
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
import os, sys
OUT = sys.argv[1]
F = os.environ.get('FONTS', 'fonts') + '/'   # SpaceGrotesk[wght].ttf, IBMPlexMono-Medium.ttf from github.com/google/fonts (OFL)
grot = instantiateVariableFont(TTFont(F + 'SpaceGrotesk[wght].ttf'), {"wght": 700})
mono = TTFont(F + 'IBMPlexMono-Medium.ttf')

def text_path(font, s, x, y, size, tracking_em):
    gs, cmap, upm = font.getGlyphSet(), font.getBestCmap(), font['head'].unitsPerEm
    sc = size / upm
    pen = SVGPathPen(gs)
    for ch in s:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (sc, 0, 0, -sc, x, y)))
        x += gs[g].width * sc + tracking_em * size
    return pen.getCommands(), x

SP = "M0 -1 C 0.1 -0.1, 0.1 -0.1, 1 0 C 0.1 0.1, 0.1 0.1, 0 1 C -0.1 0.1, -0.1 0.1, -1 0 C -0.1 -0.1, -0.1 -0.1, 0 -1 Z"
ACC, AMBER = '#0E8FB8', '#F2A93B'

def mark(ink):
    ell = [(13, 4.5, 1.6, .75), (21, 7.3, 1.9, .7), (32, 11.2, 2.3, .62), (48, 16.8, 2.8, .55), (68, 23.8, 3.3, .48)]
    beads = [(100, 5, 9.2), (114, 4.6, 8.6), (128, 4.6, 8.1), (142, 4.6, 7.6), (156, 4.6, 7.1), (170, 4.6, 6.6),
             (184, 4.6, 6.1), (198, 4.6, 5.6), (212, 4.6, 5.1), (226, 4.6, 4.6)]
    s = f'<g fill="none" stroke="{ink}" transform="rotate(-45 100 100)">'
    s += ''.join(f'<ellipse cx="100" cy="100" rx="{a}" ry="{b}" stroke-width="{w}" opacity="{o}"/>' for a, b, w, o in ell) + '</g>'
    s += f'<circle cx="100" cy="100" r="84" fill="none" stroke="{ink}" stroke-width="14"/>'
    s += f'<g fill="{ACC}" transform="rotate(45 100 100)"><path d="M28 100 C 60 99, 80 91, 100 91 L 100 109 C 80 109, 60 101, 28 100 Z" opacity="0.85"/>'
    s += ''.join(f'<ellipse cx="{c}" cy="100" rx="{a}" ry="{b}"/>' for c, a, b in beads) + '</g>'
    return s

def small_mark(ink):   # avatar geometry (fewer, larger microbunches)
    s = f'<g fill="none" stroke="{ink}" transform="rotate(-45 100 100)"><ellipse cx="100" cy="100" rx="28" ry="9.8" stroke-width="6" opacity="0.65"/><ellipse cx="100" cy="100" rx="56" ry="19.6" stroke-width="7" opacity="0.55"/></g>'
    s += f'<circle cx="100" cy="100" r="82" fill="none" stroke="{ink}" stroke-width="20"/>'
    s += f'<g fill="{ACC}" transform="rotate(45 100 100)"><path d="M28 100 C 60 98, 80 88, 100 88 L 100 112 C 80 112, 60 102, 28 100 Z"/>'
    s += ''.join(f'<ellipse cx="{c}" cy="100" rx="{a}" ry="{b}"/>' for c, a, b in [(104, 8, 12), (128, 7.5, 11), (152, 7.5, 10), (176, 7.5, 9), (200, 7.5, 8), (224, 7.5, 7)]) + '</g>'
    return s

def tiny_mark(ink):    # favicon geometry
    return (f'<circle cx="100" cy="100" r="78" fill="none" stroke="{ink}" stroke-width="28"/>'
            f'<g fill="{ACC}" transform="rotate(45 100 100)"><path d="M34 100 C 64 97, 82 84, 100 84 L 100 116 C 82 116, 64 103, 34 100 Z"/>'
            '<ellipse cx="136" cy="100" rx="12" ry="14"/><ellipse cx="178" cy="100" rx="11" ry="12"/><ellipse cx="218" cy="100" rx="10" ry="10"/></g>')

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">'
            f'<title>{title}</title>{body}</svg>\n')

def logo(dark):
    bg, ink, sub = ('#0B0F18', '#EEF1F4', '#A9B2BE') if dark else ('#F4F5F2', '#1F3A5F', '#566E8E')
    W, H = 720, 320
    b = f'<rect width="{W}" height="{H}" fill="{bg}"/>'
    b += f'<g><path d="{SP}" transform="translate(590 100) scale(150)" fill="{AMBER}" opacity="0.22"/>'
    b += f'<path d="{SP}" transform="translate(668 280) scale(22)" fill="{AMBER}" opacity="0.85"/>'
    b += f'<path d="{SP}" transform="translate(470 26) scale(11)" fill="{AMBER}" opacity="0.7"/></g>'
    b += f'<svg x="64" y="60" width="200" height="200" viewBox="-6 -6 212 212">{mark(ink)}</svg>'
    word, _ = text_path(grot, 'QUARZ', 300, 178, 92, 0.04)
    tag, _ = text_path(mono, 'QUASISTATIC ARBITRARY-RESOLUTION RZ', 302, 210, 13, 0.14)
    b += f'<path d="{word}" fill="{ink}"/><path d="{tag}" fill="{sub}"/>'
    return svg(W, H, b, 'QUARZ – Quasistatic Arbitrary-resolution RZ')

files = {
    'quarz-logo-dark.svg': logo(True),
    'quarz-logo-light.svg': logo(False),
    'quarz-mark-dark.svg': svg(212, 212, f'<g transform="translate(6 6)">{mark("#EEF1F4")}</g>', 'QUARZ'),
    'quarz-mark-light.svg': svg(212, 212, f'<g transform="translate(6 6)">{mark("#1F3A5F")}</g>', 'QUARZ'),
    'quarz-avatar.svg': svg(512, 512, f'<rect width="512" height="512" fill="#0B0F18"/><svg x="56" y="56" width="400" height="400" viewBox="-6 -6 212 212">{small_mark("#EEF1F4")}</svg>', 'QUARZ'),
    'quarz-avatar-light.svg': svg(512, 512, f'<rect width="512" height="512" fill="#F4F5F2"/><svg x="56" y="56" width="400" height="400" viewBox="-6 -6 212 212">{small_mark("#1F3A5F")}</svg>', 'QUARZ'),
    'favicon.svg': svg(32, 32, f'<rect width="32" height="32" rx="7" fill="#0B0F18"/><svg x="3" y="3" width="26" height="26" viewBox="-6 -6 212 212">{tiny_mark("#EEF1F4")}</svg>', 'QUARZ'),
}
for n, s in files.items():
    open(os.path.join(OUT, n), 'w').write(s)
print('\n'.join(files))
