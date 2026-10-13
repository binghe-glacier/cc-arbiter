"""Replace the deposited archive on Zenodo (record 23329663) with the current delivery zip.

Requires a Zenodo personal access token with deposit:read and deposit:write scopes,
supplied via the ZENODO_TOKEN environment variable. Zenodo allows replacing the files of
a published record within 30 days of publication; the DOI does not change.

Usage (from the project root, after setting ZENODO_TOKEN):
    python code/zenodo_update.py
"""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

RECORD = '23329663'
ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / 'CC-Arbiter_v3_revision_20261012.zip'
API = 'https://zenodo.org/api/records/' + RECORD


def call(method, url, data=None, ctype=None):
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        sys.exit('Set the ZENODO_TOKEN environment variable first.')
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', 'Bearer ' + token)
    if ctype:
        req.add_header('Content-Type', ctype)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main():
    if not ZIP.is_file():
        sys.exit('Delivery zip not found: ' + str(ZIP))
    body_bytes = ZIP.read_bytes()
    md5 = hashlib.md5(body_bytes).hexdigest()
    print('local zip:', ZIP.name, len(body_bytes), 'bytes, md5', md5)

    status, body = call('GET', API + '/draft')
    print('open draft (GET):', status)
    if status >= 400:
        status, body = call('POST', API + '/draft', b'')
        print('open draft (POST):', status)
    if status >= 400:
        sys.exit(body[:400])

    status, body = call('DELETE', API + '/draft/files/' + ZIP.name)
    print('remove old file:', status)

    status, body = call('POST', API + '/draft/files',
                        json.dumps([{'key': ZIP.name}]).encode(), 'application/json')
    print('create file entry:', status)
    if status >= 400:
        sys.exit(body[:400])

    status, body = call('PUT', API + '/draft/files/' + ZIP.name + '/content',
                        body_bytes, 'application/octet-stream')
    print('upload:', status)
    if status >= 400:
        sys.exit(body[:400])

    status, body = call('POST', API + '/draft/files/' + ZIP.name + '/commit')
    print('commit file:', status)
    if status >= 400:
        sys.exit(body[:400])

    status, body = call('POST', API + '/draft/actions/publish')
    print('publish:', status)
    if status >= 400:
        sys.exit(body[:400])

    with urllib.request.urlopen(API, timeout=120) as resp:
        record = json.load(resp)
    for f in record.get('files', []):
        print('record file:', f.get('key'), f.get('size'), f.get('checksum'))
    print('expected md5:', md5)
    print('done; the DOI is unchanged.')


if __name__ == '__main__':
    main()
