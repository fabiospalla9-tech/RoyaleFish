"""League decision server: answers for the two learners, their past versions and the frozen old 2.6, in big batches.
Usage: league_server.py HOST PORT"""
import json, os, pickle, queue, socket, struct, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(2); torch.distributions.Distribution.set_default_validate_args(False)
from league_common import RLB, OLD, POOL, act, cat_pad, grow

host, port = sys.argv[1], int(sys.argv[2]); dev = 'cuda'; torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEM_FRAC', '1'))); meta = json.load(open(POOL + '/meta.json'))
models = {}; stamp = {}; requests = queue.Queue(); active = [0]


def path(mid):
    return f'{POOL}/m{mid}.pt' if mid in (0, 1) else f'{POOL}/p_{mid}.pt'


def get(mid):
    if mid not in models:
        if len(models) > 16:
            models.pop(next(k for k in models if k not in (0, 1)))
        m = RLB(len(meta['vocab']), meta['units'], meta['g_w']).to(dev).half().eval(); m.load_state_dict(grow(torch.load(path(mid), map_location=dev), meta['units'], meta['grow'])); models[mid] = m; stamp[mid] = os.stat(path(mid)).st_mtime
    return models[mid]


def reload():
    for mid in (0, 1):
        try:
            t = os.stat(path(mid)).st_mtime
            if mid in models and t != stamp[mid]:
                models[mid].load_state_dict(grow(torch.load(path(mid), map_location=dev), meta['units'], meta['grow'])); stamp[mid] = t
        except (OSError, RuntimeError, EOFError):
            pass




def recv(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError
        buf += c
    return bytes(buf)


def conn(sock):
    reply = queue.Queue(); active[0] += 1
    try:
        while True:
            n = struct.unpack('!Q', recv(sock, 8))[0]; requests.put((pickle.loads(recv(sock, n)), reply))
            out = pickle.dumps(reply.get(), protocol=5); sock.sendall(struct.pack('!Q', len(out)) + out)
    except (ConnectionError, OSError):
        active[0] -= 1; sock.close()


def run_new(items):
    b = {k: np.concatenate([x[k] for x in items]) for k in ('N', 'G', 'H', 'R', 'A', 'S', 'P', 'X')}; b['E'] = cat_pad(items, 'E'); b['M'] = cat_pad(items, 'M'); n = len(b['N'])
    c = np.zeros(n, np.int64); cell = np.zeros(n, np.int64); lp = np.zeros(n, np.float32); v = np.zeros(n, np.float32)
    for mid in np.unique(b['P']).tolist():
        j = np.nonzero(b['P'] == mid)[0]; sub = {k: b[k][j] for k in ('E', 'N', 'G', 'H', 'M', 'R', 'A', 'S', 'X')}
        try:
            mdl = get(int(mid))
        except (OSError, RuntimeError, EOFError):
            mdl = get(0)
        c[j], cell[j], lp[j], v[j] = act(mdl, sub, dev)
    return c, cell, lp, v


def run_old(items):
    b = {k: np.concatenate([x[k] for x in items]) for k in ('E', 'N', 'G', 'H', 'M', 'S')}
    raise RuntimeError('the frozen older 2.6 is not published: remove a_old and b_old from MIX')
    s, ce = None, None
    return s, ce


def batcher():
    last = 0; WAIT_S = float(os.environ.get('WAIT_MS', '25')) / 1000; cnt = bat = 0; tl = time.time()
    while True:
        items = [requests.get()]; t0 = time.time()
        while len(items) < active[0] and time.time() - t0 < WAIT_S:
            try:
                items.append(requests.get_nowait())
            except queue.Empty:
                time.sleep(0.0002)
        if time.time() - last > 2:
            reload(); last = time.time()
        nw = [x[0]['new'] for x in items if x[0]['new'] is not None]; od = [x[0]['old'] for x in items if x[0]['old'] is not None]
        cnt += sum(len(x['N']) for x in nw + od); bat += 1
        if time.time() - tl > 30:
            print('decisioni/s', round(cnt / (time.time() - tl)), 'per blocco', round(cnt / max(bat, 1)), flush=True); cnt = bat = 0; tl = time.time()
        rn = run_new(nw) if nw else None; ro = run_old(od) if od else None; a = b = 0
        for x, reply in items:
            r = {'new': None, 'old': None}
            if x['new'] is not None:
                k = len(x['new']['N']); r['new'] = tuple(y[a:a + k] for y in rn); a += k
            if x['old'] is not None:
                k = len(x['old']['N']); r['old'] = tuple(y[b:b + k] for y in ro); b += k
            reply.put(r)


threading.Thread(target=batcher, daemon=True).start()
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind((host, port)); srv.listen(128); print('listening', host, port, flush=True)
while True:
    c, _ = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); threading.Thread(target=conn, args=(c,), daemon=True).start()
