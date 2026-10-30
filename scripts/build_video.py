#!/usr/bin/env python3
"""PX Video builder.

Reads a job manifest, auto-picks the best photos/clips (must-includes pinned),
transcribes the voice track with Whisper, turns the loose script into timed
on-screen graphic cards, ducks the music under the voice, and renders
16x9 + 9x16 MP4s plus an .srt and a contact sheet of what was chosen.

Usage:
  python scripts/build_video.py --manifest jobs/my-job.json \
      --media-dir ./local_media --out-dir ./out
Media files are looked up in --media-dir by file name (run fetch_media.py
first, or copy files there yourself).
"""
import argparse, difflib, json, math, os, re, subprocess, sys, textwrap
from pathlib import Path

from PIL import Image
import numpy as np

FPS = 30
IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp'}
VID_EXT = {'.mp4', '.mov', '.m4v', '.webm'}


def sh(cmd, capture=False):
    print('+', ' '.join(str(c) for c in cmd)[:240], flush=True)
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        sys.stderr.write((r.stderr or '')[-3000:])
        raise SystemExit(f"command failed: {cmd[0]}")
    return r.stdout if capture else ''


def probe(path):
    out = sh(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
              '-of', 'json', str(path)], capture=True)
    return json.loads(out)


def media_info(path):
    p = probe(path)
    dur = float(p['format'].get('duration', 0))
    w = h = None
    for s in p['streams']:
        if s['codec_type'] == 'video':
            w, h = s['width'], s['height']
    return dur, w, h


def is_image(path):
    return Path(path).suffix.lower() in IMG_EXT


# ---------------- scoring ----------------

def sharpness_of_frame(img_path):
    im = Image.open(img_path).convert('L')
    im.thumbnail((480, 480))
    a = np.asarray(im, dtype=np.float32)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    bright = float(a.mean())
    exposure_penalty = 1.0
    if bright < 35 or bright > 225:
        exposure_penalty = 0.5
    return float(lap.var()) * exposure_penalty


def score_media(path, work):
    """Return (score, best_offset_seconds)."""
    if is_image(path):
        return sharpness_of_frame(path), 0.0
    dur, _, _ = media_info(path)
    n = 5
    frames, prev, motion = [], None, 0.0
    for i in range(n):
        t = dur * (i + 0.5) / n
        fp = Path(work) / f"sc_{Path(path).stem}_{i}.jpg"
        sh(['ffmpeg', '-y', '-v', 'error', '-ss', f'{t:.2f}', '-i', str(path),
            '-frames:v', '1', str(fp)])
        frames.append((sharpness_of_frame(fp), t))
        cur = np.asarray(Image.open(fp).convert('L').resize((160, 90)), dtype=np.float32)
        if prev is not None:
            motion += float(np.abs(cur - prev).mean())
        prev = cur
    sharp = max(f[0] for f in frames)
    best_t = max(frames)[1]
    return sharp * 0.7 + motion * 30.0 * 0.3 + sharp * 0.0, best_t


# ---------------- transcription ----------------

def transcribe(voice_path, work, model_name):
    cache = Path(work) / 'transcript.json'
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("faster-whisper is not installed. Run: pip install faster-whisper")
    model = WhisperModel(model_name, device='cpu', compute_type='int8')
    # Decode with ffmpeg and hand Whisper raw samples: avoids depending on
    # PyAV's decode API, which changes between versions.
    wav16 = Path(work) / 'voice16k.wav'
    if not wav16.exists():
        sh(['ffmpeg', '-y', '-v', 'error', '-i', str(voice_path),
            '-ar', '16000', '-ac', '1', str(wav16)])
    import contextlib, wave
    with contextlib.closing(wave.open(str(wav16))) as w:
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, word_timestamps=True)
    words, segs = [], []
    for s in segments:
        segs.append({'start': s.start, 'end': s.end, 'text': s.text.strip()})
        for w in (s.words or []):
            words.append({'start': w.start, 'end': w.end, 'word': w.word})
    data = {'segments': segs, 'words': words}
    cache.write_text(json.dumps(data, indent=1))
    return data


def caption_cues(words):
    """Group word timestamps into caption cues: <=2 lines, <=42 chars/line."""
    cues, cur = [], []
    def flush():
        if not cur:
            return
        text = ''.join(w['word'] for w in cur).strip()
        lines, line = [], ''
        for word in text.split():
            if len(line) + len(word) + 1 > 42 and line:
                lines.append(line)
                line = word
            else:
                line = (line + ' ' + word).strip()
        if line:
            lines.append(line)
        cues.append({'start': cur[0]['start'], 'end': cur[-1]['end'],
                     'text': '\n'.join(lines[:2])})
        cur.clear()
    for w in words:
        if cur and (w['start'] - cur[-1]['end'] > 0.7 or
                    len(''.join(x['word'] for x in cur)) > 78):
            flush()
        cur.append(w)
        if w['word'].strip().endswith(('.', '!', '?')) and \
           len(''.join(x['word'] for x in cur)) > 30:
            flush()
    flush()
    return cues


def ts_srt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_ass(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(cues, path):
    with open(path, 'w') as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{ts_srt(c['start'])} --> {ts_srt(c['end'])}\n"
                    f"{c['text']}\n\n")


def write_ass(cues, path, brand, vertical):
    size = brand['caption_font_size_9x16'] if vertical else brand['caption_font_size_16x9']
    margin_v = 150 if vertical else 42
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {1080 if vertical else 1920}
PlayResY: {1920 if vertical else 1080}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Poppins SemiBold,{size},&H00FFFFFF,&H000000FF,&H80000000,&H96000000,1,0,0,0,100,100,0,0,3,1,0,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, 'w') as f:
        f.write(hdr)
        for c in cues:
            txt = c['text'].replace('\n', r'\N')
            f.write(f"Dialogue: 0,{ts_ass(c['start'])},{ts_ass(c['end'])},Cap,,0,0,0,,{txt}\n")


# ---------------- script cards ----------------

def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower())


def sentences_from_words(words):
    """Split the word stream into sentences with exact start times."""
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if w['word'].strip().endswith(('.', '!', '?')):
            sents.append({'text': ''.join(x['word'] for x in cur).strip(),
                          'start': cur[0]['start']})
            cur = []
    if cur:
        sents.append({'text': ''.join(x['word'] for x in cur).strip(),
                      'start': cur[0]['start']})
    return sents


def align_cards(script, segments, total, voice_end, title_floor=0.0):
    """Fuzzy-match each script line to the narration to time its card."""
    sents = [(norm(s['text']), s['start']) for s in segments]
    placed, used = [], set()
    loose = []
    for line in script:
        best, bi = 0.0, -1
        for i, (txt, _) in enumerate(sents):
            r = difflib.SequenceMatcher(None, norm(line), txt).ratio()
            if r > best:
                best, bi = r, i
        if bi >= 0 and best >= 0.30:
            placed.append({'text': line, 'start': sents[bi][1], 'matched': True})
            used.add(bi)
        else:
            loose.append(line)
    if loose:
        # drop each unmatched card into the middle of the widest free gap
        bounds = [2.8] + sorted(p['start'] for p in placed) + [max(voice_end - 3.6, 2.8)]
        for line in loose:
            gaps = [(bounds[i + 1] - bounds[i], (bounds[i] + bounds[i + 1]) / 2)
                    for i in range(len(bounds) - 1)]
            _, mid = max(gaps)
            placed.append({'text': line, 'start': mid, 'matched': False})
            bounds = sorted(bounds + [mid])
    placed.sort(key=lambda c: c['start'])
    prev_end = title_floor - 0.15 if title_floor else -1.0
    for c in placed:
        if c['start'] < prev_end + 0.15:
            c['start'] = prev_end + 0.15
        c['end'] = min(c['start'] + 3.4, voice_end)
        prev_end = c['end']
    return placed


def wrap_card(text, chars):
    return '\n'.join(textwrap.wrap(text, chars)) or text


def drawtext_escape(s):
    return (s.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
             .replace('%', '\\%'))


# ---------------- render ----------------

def seg_filter(idx, path, dur_seg, vertical, work):
    """Produce the filter fragment turning input idx into a segment stream."""
    W, H = (1080, 1920) if vertical else (1920, 1080)
    _, sw, shgt = media_info(path)
    src_ar = (sw / shgt) if sw and shgt else 16 / 9
    tgt_ar = W / H
    label = f"v{idx}"
    if abs(src_ar - tgt_ar) < 0.12:
        base = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    else:
        base = (f"split[a{idx}][b{idx}];[a{idx}]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},gblur=sigma=42,eq=brightness=-0.12[bgb{idx}];"
                f"[b{idx}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{idx}];"
                f"[bgb{idx}][fg{idx}]overlay=(W-w)/2:(H-h)/2")
    if is_image(path):
        # gentle Ken Burns on stills
        return (f"[{idx}:v]scale={int(W*1.25)}:{int(H*1.25)}:force_original_aspect_ratio=increase,"
                f"crop={int(W*1.25)}:{int(H*1.25)},zoompan=z='min(zoom+0.0009,1.09)':d={int(dur_seg*FPS)}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS},"
                f"trim=duration={dur_seg:.2f},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[{label}]")
    return f"[{idx}:v]{base},fps={FPS},trim=duration={dur_seg:.2f},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[{label}]"


def render(manifest, media, plan, cards, cues, brand, work, out_path, vertical, preset):
    W, H = (1080, 1920) if vertical else (1920, 1080)
    inputs, filters = [], []
    for i, seg in enumerate(plan):
        p = seg['path']
        if is_image(p):
            inputs += ['-loop', '1', '-t', f"{seg['dur']:.2f}", '-i', str(p)]
        else:
            inputs += ['-stream_loop', '-1', '-ss', f"{seg['offset']:.2f}",
                       '-t', f"{seg['dur']:.2f}", '-i', str(p)]
        filters.append(seg_filter(i, p, seg['dur'], vertical, work))
    voice_i = len(plan)
    music_i = len(plan) + 1
    inputs += ['-i', str(media['voice']), '-stream_loop', '-1', '-i', str(media['music'])]
    total = plan[-1]['end']
    n = len(plan)
    graph = ';'.join(filters) + ';' + ''.join(f"[v{i}]" for i in range(n)) + \
        f"concat=n={n}:v=1:a=0[cat]"
    # graphic cards as drawtext overlays
    fonts = Path(__file__).resolve().parent.parent / 'assets' / 'fonts'
    semi = fonts / brand['font_semibold']
    bold = fonts / brand['font_bold']
    accent = brand['accent_hex']
    fs_card = 54 if vertical else 58
    fs_title = 76 if vertical else 84
    chars = 24 if vertical else 34
    x = '70' if not vertical else '(w-text_w)/2'
    y_card = '120' if not vertical else '150'
    dt = []
    card_no = [0]
    def card(text, start, end, size, yy, centered=False):
        xx = '(w-text_w)/2' if centered else x
        card_no[0] += 1
        tf = Path(work) / f"card_{'v' if vertical else 'h'}_{card_no[0]}.txt"
        tf.write_text(text)
        return (f"drawtext=fontfile='{bold}':textfile='{tf}':fontsize={size}:"
                f"fontcolor=white:borderw=0:box=1:boxcolor=black@{brand['card_bg_alpha']}:"
                f"boxborderw=26:x={xx}:y={yy}:"
                f"alpha='if(lt(t,{start:.2f}),0,if(lt(t,{start+0.35:.2f}),(t-{start:.2f})/0.35,"
                f"if(lt(t,{end-0.35:.2f}),1,if(lt(t,{end:.2f}),({end:.2f}-t)/0.35,0))))':"
                f"enable='between(t,{start:.2f},{end:.2f})'")
    title = manifest.get('title', '')
    if title:
        dt.append(card(wrap_card(title.upper(), 20 if vertical else 26), 0.3, 2.9, fs_title, y_card))
    for c in cards:
        dt.append(card(wrap_card(c['text'], chars), c['start'], c['end'], fs_card, y_card))
    end_txt = brand.get('end_card_text', '')
    if end_txt:
        dt.append(card(wrap_card(end_txt, 22 if vertical else 30), total - 2.4, total - 0.1,
                       fs_card, '(h-text_h)/2', centered=True))
    vid = '[cat]' + ','.join(dt) + '[withcards]'
    ass_path = Path(work) / ('captions_9x16.ass' if vertical else 'captions_16x9.ass')
    write_ass(cues, ass_path, brand, vertical)
    fontsdir = str(fonts).replace(':', '\\:')
    graph += f";{vid};[withcards]subtitles='{str(ass_path).replace(':', chr(92)+':')}':fontsdir='{fontsdir}'[vout]"
    # audio: voice loudnorm, music ducked by voice, fades
    graph += (f";[{voice_i}:a]loudnorm=I=-16:TP=-1.5:LRA=11,asplit=2[vox][side];"
              f"[vox]atrim=0:{total:.2f},asetpts=PTS-STARTPTS[voxT];"
              f"[{music_i}:a]atrim=0:{total:.2f},volume=0.55[mus];"
              f"[mus][side]sidechaincompress=threshold=0.03:ratio=8:attack=200:release=1000[duck];"
              f"[duck]afade=t=in:st=0:d=0.6,afade=t=out:st={max(total-2.5,0):.2f}:d=2.5[duckF];"
              f"[voxT][duckF]amix=inputs=2:duration=first:dropout_transition=0,"
              f"alimiter=limit=0.95[aout]")
    (Path(work) / 'filtergraph.txt').write_text(graph)
    cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', graph,
           '-map', '[vout]', '-map', '[aout]', '-c:v', 'libx264', '-preset', preset,
           '-crf', '19', '-pix_fmt', 'yuv420p', '-r', str(FPS),
           '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-t', f'{total:.2f}',
           str(out_path)]
    sh(cmd)


def choose_segments(manifest, media_dir, work, total):
    pool = []
    for name in manifest.get('media', []):
        p = Path(media_dir) / Path(name).name
        if p.exists():
            pool.append(p)
    must = []
    for name in manifest.get('must_include', []):
        p = Path(media_dir) / Path(name).name
        if p.exists() and p not in must:
            must.append(p)
    scored = []
    for p in pool:
        s, off = score_media(p, work)
        scored.append({'path': p, 'score': s, 'offset': off})
        print(f"  score {p.name}: {s:.0f}")
    scored.sort(key=lambda d: -d['score'])
    n_seg = max(2, round(total / 4.2))
    chosen = []
    for p in must:
        match = next((d for d in scored if d['path'] == p), {'path': p, 'score': 0, 'offset': 0})
        chosen.append({**match, 'must': True})
    for d in scored:
        if len(chosen) >= n_seg:
            break
        if d['path'] not in [c['path'] for c in chosen]:
            chosen.append({**d, 'must': False})
    # variety: alternate clips and stills where possible
    chosen.sort(key=lambda d: (not d.get('must', False), is_image(d['path'])))
    plan, t = [], 0.0
    per = total / len(chosen)
    for i, d in enumerate(chosen):
        end = total if i == len(chosen) - 1 else t + per
        plan.append({'path': d['path'], 'start': t, 'end': end,
                     'dur': end - t, 'offset': d['offset'], 'score': d['score']})
        t = end
    return plan


def contact_sheet(plan, work, out_path):
    tiles = []
    for i, seg in enumerate(plan):
        fp = Path(work) / f"tile_{i}.jpg"
        src = seg['path']
        if is_image(src):
            ss = []
        else:
            clip_dur, _, _ = media_info(src)
            ss = ['-ss', f"{max(0.0, min(seg['offset'] + seg['dur'] / 2, clip_dur - 0.2)):.2f}"]
        sh(['ffmpeg', '-y', '-v', 'error'] + ss + ['-i', str(src), '-frames:v', '1',
           '-vf', 'scale=480:270:force_original_aspect_ratio=increase,crop=480:270', str(fp)])
        tiles.append(Image.open(fp))
    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new('RGB', (cols * 480, rows * 270), 'black')
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 480, (i // cols) * 270))
    sheet.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--media-dir', default='./local_media')
    ap.add_argument('--out-dir', default='./out')
    ap.add_argument('--whisper-model', default=os.environ.get('WHISPER_MODEL', 'base'))
    ap.add_argument('--preset', default='veryfast')
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    repo = Path(__file__).resolve().parent.parent
    brand = json.load(open(repo / 'assets' / 'brand.json'))
    job_id = manifest.get('id') or Path(args.manifest).stem
    work = Path(args.out_dir) / job_id / 'work'
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) / job_id
    media_dir = Path(args.media_dir)
    media = {'voice': media_dir / Path(manifest['voice']).name,
             'music': media_dir / Path(manifest['music']).name}
    for k, p in media.items():
        if not p.exists():
            raise SystemExit(f"{k} file not found in {media_dir}: {p.name} "
                             f"(run scripts/fetch_media.py first)")

    voice_dur, _, _ = media_info(media['voice'])
    total = voice_dur + 1.2
    print(f"Voice: {voice_dur:.1f}s -> video length {total:.1f}s")

    print("Transcribing voice track...")
    tr = transcribe(media['voice'], work, args.whisper_model)
    cues = caption_cues(tr['words']) if tr['words'] else []
    write_srt(cues, out_dir / f"{job_id}.srt")

    print("Scoring media pool...")
    plan = choose_segments(manifest, media_dir, work, total)

    sents = sentences_from_words(tr['words']) or tr['segments']
    cards = align_cards(manifest.get('script', []), sents, total, voice_dur,
                        title_floor=3.1 if manifest.get('title') else 0.0)
    for c in cards:
        print(f"  card @{c['start']:.1f}s ({'matched' if c['matched'] else 'placed'}): {c['text'][:60]}")

    json.dump({'segments': [{**{k: (str(v) if k == 'path' else v) for k, v in s.items()}}
                             for s in plan], 'cards': cards},
              open(out_dir / 'plan.json', 'w'), indent=1)
    contact_sheet(plan, work, out_dir / 'contact-sheet.jpg')

    render(manifest, media, plan, cards, cues, brand, work,
           out_dir / f"{job_id}_16x9.mp4", vertical=False, preset=args.preset)
    render(manifest, media, plan, cards, cues, brand, work,
           out_dir / f"{job_id}_9x16.mp4", vertical=True, preset=args.preset)
    print(f"Done: {out_dir}")


if __name__ == '__main__':
    main()
