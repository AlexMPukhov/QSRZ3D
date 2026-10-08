# QUARZ logo

A self-modulated proton bunch crossing the code's radial grid. The ring is the Q; inside it,
the stretched radial grid (rings fine at the axis, coarse outside) drawn as a tilted disk,
the quasi-3D (r, θ, ξ) picture. The bunch is thickest where it crosses the grid and breaks
into microbunches behind it, as in AWAKE; the last ones form the tail of the Q. The amber
sparks in the background mark that the code was developed with an AI assistant (Claude).

| file | use |
|---|---|
| `quarz-logo-dark.svg` / `.png` | full logo on a dark background (README in dark mode, slides) |
| `quarz-logo-light.svg` / `.png` | full logo on a light background (README, posters, papers) |
| `quarz-mark-dark.svg` / `-light.svg` (`.png`) | the Q alone, transparent background |
| `quarz-avatar.svg` / `.png` | 512 × 512, simplified mark: GitHub avatar, social media |
| `favicon.svg` | 32 × 32, ring, head and three microbunches |

Colours: background `#0B0F18` (dark) / `#F4F5F2` (light), ink `#EEF1F4` / navy `#1F3A5F` (tagline `#566E8E`), bunch
cyan `#0E8FB8`, sparks amber `#F2A93B`. Lettering: Space Grotesk Bold (wordmark) and
IBM Plex Mono Medium (tagline), both SIL Open Font License, converted to outlines, so the
SVGs need no fonts. `make_logo.py` regenerates the SVGs
(`FONTS=<dir> python3 make_logo.py <out dir>`); the PNGs are rendered from them at 2×.
