"""Exact frozen R source-packing helpers; no planner, teacher or labels."""
from datetime import datetime, timezone
import tiktoken
ENC=tiktoken.get_encoding("o200k_base")
def tokens(text): return len(ENC.encode(text, disallowed_special=()))

def render_source(source,content=None):
    stamp=source.get('timestamp');date='unknown' if stamp is None else datetime.fromtimestamp(stamp/1000,timezone.utc).isoformat()
    return f"[source={source['id']}; role={source['role']}; timestamp={date}]\n"+(source['content'] if content is None else content)

def pack(ids,sources,token_budget=16000,max_items=100):
    output=[];seen=set();used=0;trace=[]
    for sid in ids:
        if sid in seen:trace.append({'id':sid,'reason':'duplicate'});continue
        seen.add(sid)
        if sid not in sources:raise ValueError('unbound evidence source')
        if len(output)>=max_items:trace.append({'id':sid,'reason':'item_budget'});continue
        source=sources[sid];text=render_source(source);length=tokens(text)
        if used+length>token_budget:
            # Whole-item omission, same deterministic policy across arms.
            trace.append({'id':sid,'reason':'token_budget','tokens':length});continue
        output.append({'id':sid,'content':text,'source_ids':[sid],'kind':'source','score':float(max_items-len(output))})
        used+=length;trace.append({'id':sid,'reason':'included','tokens':length,'original_message_index':source['original_message_index'],'char_span':source['char_span']})
    return {'evidence':output,'selection_trace':trace,'evidence_tokens':used}

def neighbors(ids,ordered,limit=8):
    by={r['id']:i for i,r in enumerate(ordered)};seen=set(ids);extra=[]
    for sid in ids:
        i=by[sid]
        for j in [i-1,i+1]:
            if 0<=j<len(ordered) and ordered[j]['session_id']==ordered[i]['session_id'] and ordered[j]['id'] not in seen:
                extra.append(ordered[j]['id']);seen.add(ordered[j]['id'])
                if len(extra)>=limit:return extra
    return extra
