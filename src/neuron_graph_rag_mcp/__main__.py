import os
import sys

service_log = os.environ.pop("_NGR_MCP_SERVICE_LOG", None)
os.environ.pop("_NGR_MCP_SERVICE_COMMAND", None)
os.environ.pop("_NGR_MCP_SERVICE_CWD", None)
if service_log:
    stream = open(service_log, "a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream

from .server import main

main()
