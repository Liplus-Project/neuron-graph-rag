import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--migrate-home":
    from neuron_graph_rag.home_migration import main

    main(sys.argv[2:])
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    from importlib.metadata import version

    print(version("neuron-graph-rag"))
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "--configure-clients":
    from .native_registration import main

    main()
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "--tray-controller":
    from .tray_controller import main

    main()
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "--tray-action":
    from .tray_controller import action_main

    action_main(sys.argv[2:])
    raise SystemExit(0)

service_log = os.environ.pop("_NGR_MCP_SERVICE_LOG", None)
os.environ.pop("_NGR_MCP_SERVICE_COMMAND", None)
os.environ.pop("_NGR_MCP_SERVICE_CWD", None)
if service_log:
    stream = open(service_log, "a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream

from .server import main

main()
