"""Test helper: the start line as the examiner (or the start camera) works it now: every runner
lined up gets their own GO, one after another."""


def start_lined_up(client, countdown_s: int = 0):
    """Start everyone at the line, one by one; returns the last start's response (None if nobody)."""
    r = None
    for e in client.get("/api/entries", params={"status": "lined_up"}).json():
        r = client.post(f"/api/entries/{e['id']}/start", json={"countdown_s": countdown_s})
        assert r.status_code == 200, r.text
    return r
