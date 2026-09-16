import json, os, glob, struct, sys
from wandb.proto import wandb_internal_pb2 as pb

BLOCK=32768; WANDB_HDR=7
def records(path):
    d=open(path,'rb').read()
    blocks=[]
    start=0
    while start < len(d):
        blocks.append((start, min(start+BLOCK, len(d))))
        start+=BLOCK
    pending=b''
    for bstart,bend in blocks:
        off = bstart + (WANDB_HDR if bstart==0 else 0)
        while off + 7 <= bend:
            crc,ln,typ = struct.unpack('<IHB', d[off:off+7])
            off+=7
            remaining = bend-off
            if ln==0 and typ==0: break
            if ln > remaining:
                off = bend; break
            data = d[off:off+ln]; off+=ln
            if typ==1: yield data
            elif typ==2: pending=data
            elif typ==3: pending+=data
            elif typ==4:
                pending+=data; yield pending; pending=b''
        # skip trailing zeros/padding to block end
    if pending: yield pending

for d in sorted(glob.glob('/data/lz/MemoryQwen/wandb/run-*')):
    f=glob.glob(os.path.join(d,'*.wandb'))
    if not f: continue
    f=f[0]
    print('===', os.path.basename(d), os.path.getsize(f), flush=True)
    n=h=0
    for data in records(f):
        rec=pb.Record(); 
        try: rec.ParseFromString(data)
        except Exception as e:
            print('  parsefail', e); continue
        which=rec.WhichOneof('record_type'); n+=1
        if which=='history':
            item={i.key:(json.loads(i.value_json) if i.value_json else None) for i in rec.history.item}
            h+=1
            print('  H', rec.history.step, json.dumps(item, sort_keys=True), flush=True)
        elif which=='summary':
            item={i.key:(json.loads(i.value_json) if i.value_json else None) for i in rec.summary.update}
            print('  S', json.dumps(item, sort_keys=True), flush=True)
    print('  records:',n,'hist:',h, flush=True)
