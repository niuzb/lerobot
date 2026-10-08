"""Loopback relay on the Merlin SSH host; target is the current GPU worker."""
import socket
import socketserver
import sys
import threading


def copy_stream(source, destination):
    try:
        while data := source.recv(1024 * 1024):
            destination.sendall(data)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection((sys.argv[1], 50055), timeout=10) as upstream:
            upstream.settimeout(None)
            reverse = threading.Thread(target=copy_stream, args=(upstream, self.request), daemon=True)
            reverse.start()
            copy_stream(self.request, upstream)
            reverse.join()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", 50056), Relay) as server:
        print(f"RELAY_READY 127.0.0.1:50056 -> [{sys.argv[1]}]:50055", flush=True)
        server.serve_forever()
