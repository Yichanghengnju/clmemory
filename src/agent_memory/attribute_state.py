"""Conservative, source-linked self preference updates. No generated user facts.

Only standalone first-person updates are parsed. Unknown referents stay unknown;
quoted, third-person and assistant instructions cannot change user state. The
original source remains the deliverable record, including negation and time.
"""
import re

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = {'i','my','me','that','the','a','an','to','of','for','and','in','is','as',
         'like','likes','love','loves','enjoy','enjoys','prefer','prefers','preference',
         'food','cuisine','cuisines','using','use','please','forget','now','instead'}
_WITHDRAW = re.compile(r"^(?:please\s+)?forget\s+(?:that\s+)?I\s+(?:like|love|enjoy|prefer)\s+(.+?)[.!]?$", re.I)
_CN_WITHDRAW = re.compile(r"^(?:请)?忘记我(?:喜欢|偏好)(.+?)[。！]?$")
_SLOT = re.compile(r"^(?:(?:correction|update):\s*)?my\s+(?:preferred|favorite|favourite)\s+([\w -]{2,50}?)\s+is\s+(.+?)[.!]?$", re.I)
_CHANGE = re.compile(r"^(?:(?:correction|update):\s*)?I\s+(?:now\s+)?prefer\s+(.+?)\s+instead of\s+(.+?)[.!]?$", re.I)
_NEGATIVE = re.compile(r"^I\s+no longer\s+(?:like|love|enjoy|prefer)\s+(.+?)[.!]?$", re.I)
_UNSAFE = re.compile(r"password|passcode|social security|ssn|email address|phone number|home address|密码|身份证|住址", re.I)


def terms(text):
    return {w.casefold() for w in _WORDS.findall(text)} - _STOP


def _mentions(text, value):
    # Examples and the compared alternative are not part of the withdrawn value.
    core = re.split(r"\b(?:such as|including|more than|rather than|for main)\b", value, maxsplit=1, flags=re.I)[0].strip()
    if re.search(r'[\u3400-\u9fff]',core):
        return core.rstrip('。！.!') in text
    wanted = terms(core)
    return bool(wanted) and wanted <= terms(text)


def _subject_source(s):
    if s['role']=='assistant':
        return bool(re.search(r'\byou\s+(?:(?:previously\s+)?(?:said|mentioned|told|enjoy|like|love|prefer|rely|use|favor|favour))\b|\byour\s+(?:preference|love|interest|favorite|favourite)\b|你(?:喜欢|偏好)',s['content'],re.I))
    return bool(re.search(r'\b(?:I\s+(?:like|love|enjoy|prefer)|my\s+(?:preferred|favorite|favourite))\b|我(?:喜欢|偏好)',s['content'],re.I))


def preference_events(user_id, sources):
    """Return exact-source updates and prior/stale source dependencies.

    Sources are supplied in the store's chronological order. A later explicit
    user assertion can establish a new value; assistant copies cannot do so.
    """
    events, blocked, prior, active = [], {}, [], []
    for s in sources:
        text = s['content'].strip()
        event = None
        if s['role'] == 'user' and '\n' not in text:
            match = _WITHDRAW.fullmatch(text) or _CN_WITHDRAW.fullmatch(text)
            slot = _SLOT.fullmatch(text)
            change = _CHANGE.fullmatch(text)
            negative = _NEGATIVE.fullmatch(text)
            if match and not _UNSAFE.search(match[1]):
                event = dict(kind='withdrawn_no_opposite_inference', attribute='preference', value=match[1])
            elif negative and not _UNSAFE.search(negative[1]):
                event = dict(kind='explicit_no_longer', attribute='preference', value=negative[1])
            elif change and not _UNSAFE.search(text):
                event = dict(kind='corrected', attribute='preference', value=change[2], new_value=change[1])
            elif slot and not _UNSAFE.search(text):
                earlier = [p for p in prior if p['role']=='user' and
                           (m := _SLOT.fullmatch(p['content'].strip())) and m[1].casefold()==slot[1].casefold()]
                if earlier:
                    old = _SLOT.fullmatch(earlier[-1]['content'].strip())
                    if old[2].casefold() != slot[2].casefold():
                        event = dict(kind='corrected', attribute=slot[1].casefold(), value=old[2], new_value=slot[2])
        if event:
            linked = [p['id'] for p in prior if _mentions(p['content'], event['value']) and _subject_source(p)]
            event.update(subject=user_id, source_id=s['id'], timestamp=s.get('timestamp'),
                         sequence=s.get('seq'), linked_source_ids=linked)
            events.append(event); active.append(event)
            for sid in linked:
                blocked[sid] = 'superseded_attribute:' + s['id']
        else:
            for e in list(active):
                # Reinstatement needs an explicit standalone user assertion.
                if s['role']=='user' and re.fullmatch(r"I (?:like|love|enjoy|prefer) .+[.!]?", text, re.I) and _mentions(text,e['value']):
                    active.remove(e)
                elif s['role']=='assistant' and _subject_source(s) and _mentions(text,e['value']):
                    blocked[s['id']] = 'stale_attribute_copy:' + e['source_id']
        prior.append(s)
    return events, blocked


def relevant_events(events, query, source_ids=()):
    ids = set(source_ids)
    q = terms(query)
    return [e for e in events if ids.intersection(e['linked_source_ids']) or
            bool(q.intersection(terms(e['value']) | terms(e['attribute']) | terms(e.get('new_value',''))))]
