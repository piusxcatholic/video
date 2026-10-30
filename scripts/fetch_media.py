#!/usr/bin/env python3
"""Download every file a manifest references into a local media folder.

Sources per manifest entry:
  - plain path that exists locally      -> copied
  - http(s) URL                          -> downloaded
  - any other path                       -> Bunny storage object:
        https://{BUNNY_STORAGE_HOST}/{ZONE}/{path}
        with header  AccessKey: {BUNNY_STORAGE_API_KEY}
    NOTE: the key must be the storage ZONE's own password (Bunny dashboard >
    Storage > your zone > FTP & API Access), NOT your Bunny account API key.
    The account key gets a 401 here - that was the old pipeline's bug.
    BUNNY_STORAGE_HOST defaults to storage.bunnycdn.com; set it to the
    regional host shown on the zone page if yours differs (e.g. uk.storage...).
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import requests


def fetch_one(ref, out_dir, zone, key, host):
    name = Path(ref.split('?')[0]).name
    dest = Path(out_dir) / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {name}")
        return dest
    if os.path.exists(ref):
        shutil.copy(ref, dest)
        print(f"  local   {name}")
    elif ref.startswith(('http://', 'https://')):
        r = requests.get(ref, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print(f"  http    {name}")
    else:
        if not zone or not key:
            sys.exit(f"Bunny path '{ref}' needs BUNNY_STORAGE_ZONE and "
                     f"BUNNY_STORAGE_API_KEY (the zone password) in the environment.")
        url = f"https://{host}/{zone}/{ref.lstrip('/')}"
        r = requests.get(url, headers={'AccessKey': key}, stream=True, timeout=120)
        if r.status_code == 401:
            sys.exit("Bunny said 401 Unauthorized. The AccessKey must be the storage "
                     "ZONE's password (Storage > zone > FTP & API Access), not the "
                     "account API key. Also check BUNNY_STORAGE_HOST matches the "
                     "regional host on that same page.")
        r.raise_for_status()
        with open(dest, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print(f"  bunny   {name}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', default='./local_media')
    args = ap.parse_args()
    m = json.load(open(args.manifest))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    zone = os.environ.get('BUNNY_STORAGE_ZONE', '')
    key = os.environ.get('BUNNY_STORAGE_API_KEY', '')
    host = os.environ.get('BUNNY_STORAGE_HOST', 'storage.bunnycdn.com')
    refs = [m['voice'], m['music']] + list(m.get('media', []))
    for ref in refs:
        fetch_one(ref, args.out_dir, zone, key, host)
    print(f"Done: {len(refs)} files in {args.out_dir}")


if __name__ == '__main__':
    main()
