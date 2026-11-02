"""How the camera processes on the station PC reach the station app.

The app may serve HTTPS (a certificate from TestPoint's own local authority, so other devices
on the LAN can use their cameras in the browser). The camera processes always talk to it on
this PC's loopback address, where checking that certificate adds nothing: for ``https`` to
127.0.0.1 / localhost they skip the check; any other address is verified as usual.
"""

from __future__ import annotations

import ssl
import urllib.parse

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def ssl_context(url: str) -> ssl.SSLContext | None:
    """The TLS context for a request to ``url``: None (the default, verified) unless it is HTTPS
    to this PC's loopback address."""
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or u.hostname not in LOOPBACK:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
