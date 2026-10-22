"""确认列表是有界展示，决议是按主键关联的事实；不改写历史交易回执。"""
from datetime import datetime, timezone


def recent_confirmations(previous, incoming, *, now=None):
    entries = {item['confirmation_id']: item for item in previous}
    for item in incoming:
        old = entries.get(item['confirmation_id'])
        if old is None or old['status'] == 'pending':
            entries[item['confirmation_id']] = item
    now = now or datetime.now(timezone.utc)
    ordered = sorted(entries.values(), key=lambda c: (datetime.fromisoformat(c['created_at']), c['confirmation_id']), reverse=True)
    active = [c for c in ordered if c['status'] == 'pending' and not c['expired']
              and datetime.fromisoformat(c['expires_at']) > now]
    active_ids = {c['confirmation_id'] for c in active}
    history = [c for c in ordered if c['confirmation_id'] not in active_ids][:20]
    selected = active_ids | {c['confirmation_id'] for c in history}
    return [c for c in ordered if c['confirmation_id'] in selected]


def confirmation_references(presentations):
    return list(dict.fromkeys(p['payload']['confirmation']['confirmation_id']
        for p in presentations if p['kind'] == 'order' and p['payload'].get('confirmation')))
