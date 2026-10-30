# PX Video — clean rebuild

Turns photos, clips, a voice track, a music track, and a loose written script
into a finished social video: **16x9 MP4 + 9x16 MP4 + an .srt caption file**,
with burned-in captions, on-screen graphic cards from your script, the music
ducked under the voice, and your branding from `assets/brand.json`.

## How a job flows

1. Upload the raw assets (photos, clips, voice, music) to the Bunny storage zone.
2. Add a job file in `jobs/` (copy `jobs/example-classroom.json`).
   - `media` = the pool the builder may pick from. It scores every photo/clip
     for sharpness (and motion, for clips) and fills the video's length with
     the best ones.
   - `must_include` = files that always appear, no matter the score.
   - `script` = your loose summary. Each line becomes a graphic card on screen,
     timed by matching its words to the narration (Whisper transcript), so the
     script does **not** need to be word-for-word.
   - `voice` = the narration. It sets the video's length and becomes the
     burned-in captions automatically.
3. Run it: **Actions → Render Video → Run workflow**, type the job file path
   (e.g. `jobs/my-video.json`), Run.
4. Download the results from the run's **Artifacts** (`finished-video`):
   both MP4s, the .srt, a `contact-sheet.jpg` showing which shots were picked,
   and `plan.json` with the scores and card timings.

Rendering happens on GitHub's machines, so long renders don't tie up yours.
On a public repo, standard Actions minutes are free; on a private repo the
free tier's 2,000 minutes a month covers a lot of 1-minute videos.

## Secrets (Settings → Secrets and variables → Actions)

| Secret | What it is |
|---|---|
| `BUNNY_STORAGE_ZONE` | The storage zone's name |
| `BUNNY_STORAGE_API_KEY` | **The storage zone's own password** — Bunny dashboard → Storage → your zone → **FTP & API Access**. NOT the account API key from account settings; the account key gets a 401. |
| `BUNNY_STORAGE_HOST` | Optional. Only if your zone page shows a regional host (e.g. `uk.storage.bunnycdn.com`). Default: `storage.bunnycdn.com` |

## Layout

```
.github/workflows/render-video.yml   the only workflow, in the only place Actions looks
assets/brand.json                    colors, fonts, school name, end-card text
assets/fonts/                        Poppins (Regular / SemiBold / Bold)
scripts/fetch_media.py               downloads a job's assets (Bunny / URL / local)
scripts/build_video.py               scores, transcribes, aligns, renders
jobs/                                one JSON file per video
```

Keep one copy of each file — the copies inside these folders. Loose duplicates
at the repo root are what made the old repo confusing.

## Test locally

```
pip install faster-whisper pillow numpy requests
python scripts/fetch_media.py --manifest jobs/example-classroom.json --out-dir ./local_media
python scripts/build_video.py --manifest jobs/example-classroom.json --media-dir ./local_media --out-dir ./out
```

## Notes and limits

- Captions are burned in (best for social feeds, where most viewing is muted)
  and the same captions ship as an .srt for YouTube / the website player.
- 9x16 rule: shots that are nearly vertical crop full-frame; landscape shots
  sit on a blurred copy of themselves so nothing important is cut off.
- If a script line can't be matched to the narration, its card is placed in a
  free gap instead — check `plan.json`'s `matched` flags if a card lands oddly.
