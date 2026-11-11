"""Length-prefixed pickle messages over TCP. Use only on a trusted local network (pickle is not safe against untrusted peers)."""
import pickle, struct


def recv_exact(sock, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError('closed')
        got += k
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack('>I', recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, n))


def send_msg(sock, obj, lock=None):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    payload = struct.pack('>I', len(data)) + data
    if lock is not None:
        with lock:
            sock.sendall(payload)
    else:
        sock.sendall(payload)
