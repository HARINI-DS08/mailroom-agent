"""Ed25519 signature verification, RFC 8032.

Uses `cryptography` when it is installed and falls back to a self-contained
implementation otherwise, so the service never depends on a wheel being
available on the host. Both paths are checked against the same signatures in
the tests, and only `verify()` is public.
"""
import hashlib

try:  # pragma: no cover - exercised by whichever path the host provides
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    def _verify_fast(public_key, signature, message):
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
            return True
        except (_InvalidSignature, ValueError):
            return False
except Exception:  # pragma: no cover
    _verify_fast = None


# --------------------------------------------------------- pure python path

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _recover_x(y, sign):
    if y >= _P:
        return None
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if xx == 0:
        return None if sign else 0
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = x * _I % _P
    if (x * x - xx) % _P != 0:
        return None
    if x & 1 != sign:
        x = _P - x
    return x


# Points are extended homogeneous coordinates (X, Y, Z, T) with x=X/Z, y=Y/Z.
_G_Y = 4 * pow(5, _P - 2, _P) % _P
_G_X = _recover_x(_G_Y, 0)
_G = (_G_X, _G_Y, 1, _G_X * _G_Y % _P)
_ZERO = (0, 1, 1, 0)


def _add(p, q):
    a, b, c, d = p
    e, f, g, h = q
    r = (b - a) * (f - e) % _P
    s = (b + a) * (f + e) % _P
    t = 2 * d * h * _D % _P
    u = 2 * c * g % _P
    return ((s - r) * (u - t) % _P, (s + r) * (u + t) % _P,
            (u + t) * (u - t) % _P, (s - r) * (s + r) % _P)


def _mul(point, scalar):
    result = _ZERO
    while scalar > 0:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _equal(p, q):
    a, b, c, _ = p
    e, f, g, _ = q
    return (a * g - e * c) % _P == 0 and (b * g - f * c) % _P == 0


def _decompress(data):
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _verify_pure(public_key, signature, message):
    if len(public_key) != 32 or len(signature) != 64:
        return False
    a = _decompress(public_key)
    if a is None:
        return None if False else False
    r = _decompress(signature[:32])
    if r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(), "little") % _L
    return _equal(_mul(_G, s), _add(r, _mul(a, h)))


def verify(public_key: bytes, signature: bytes, message: bytes) -> bool:
    """True when `signature` is a valid Ed25519 signature over `message`."""
    if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != 32:
        return False
    if not isinstance(signature, (bytes, bytearray)) or len(signature) != 64:
        return False
    if _verify_fast is not None:
        return _verify_fast(bytes(public_key), bytes(signature), bytes(message))
    return _verify_pure(bytes(public_key), bytes(signature), bytes(message))
