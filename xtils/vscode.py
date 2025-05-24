import json
import os
import socket


def _make_vscode_debug_launcher_uri(port: int):
    launch_config = {
        "name": "Python: Remote Attach",
        "type": "debugpy",
        "request": "attach",
        "presentation": {"hidden": True},
        "purpose": ["debug-in-terminal"],
        "connect": {"host": "localhost", "port": port},
        "pathMappings": [{"localRoot": "${workspaceFolder}", "remoteRoot": os.getcwd()}],
        "justMyCode": True,
        "console": "integratedTerminal",
        "internalConsoleOptions": "neverOpen",
        "redirectOutput": "false",
        "autoReload": {"enable": False},
    }
    launch_config = json.dumps(launch_config)
    return f"vscode://fabiospampinato.vscode-debug-launcher/launch?args={launch_config}"


def _handle_vscode_remote(vscode_ipc: str, port: int) -> None:
    # the VSCode Remote extension does not support `code --open-url {url}` with a `vscode://` extension
    # This may change in the future, but for now we need to bypass this limitation by using the VSCode IPC
    # secket to send the `vscode://` url to the VSCode instance server directly
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.connection import HTTPConnection
    from urllib3.connectionpool import HTTPConnectionPool

    class VSCodeIPCConnection(HTTPConnection):
        def __init__(self):
            super().__init__("localhost")

        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(vscode_ipc)

    class VSCodeIPCConnectionPool(HTTPConnectionPool):
        def __init__(self):
            super().__init__("localhost")

        def _new_conn(self):  # type: ignore
            return VSCodeIPCConnection()

    class VSCodeIPCAdapter(HTTPAdapter):
        def get_connection(self, url, proxies=None):
            del url, proxies
            return VSCodeIPCConnectionPool()

        def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
            del request, verify, proxies, cert
            return VSCodeIPCConnectionPool()

    session = requests.Session()
    session.mount("vscode://", VSCodeIPCAdapter())
    session.post(
        "vscode://",
        headers={"content-type": "application/json", "accept": "application/json"},
        json={
            "type": "openExternal",
            "uris": [_make_vscode_debug_launcher_uri(port)],
        },
    )


def _handle_vscode_local(port: int) -> None:
    import subprocess

    subprocess.run(["code", "--open-url", _make_vscode_debug_launcher_uri(port)])


def attach_to_debugpy() -> None:
    os.environ["PYTHONBREAKPOINT"] = "debugpy.breakpoint"

    import debugpy
    import portpicker  # type: ignore

    # Find an open port to listen on
    port = portpicker.pick_unused_port()
    debugpy.listen(port)

    # If we're in a local vscode terminal session, we need to tell VSCode to connect to the debug adapter
    # using fabiospampinato.vscode-debug-launcher extension
    if is_vscode_tty():
        if vscode_ipc := os.environ.get("VSCODE_IPC_HOOK_CLI", None):
            _handle_vscode_remote(vscode_ipc, port)
        else:
            _handle_vscode_local(port)
    else:
        print(f"Waiting for debugger to attach on port {port}...")

    debugpy.wait_for_client()


def is_vscode_tty() -> bool:
    return os.environ.get("TERM_PROGRAM", None) == "vscode" and os.environ.get("TERM_PROGRAM_VERSION", None) is not None
