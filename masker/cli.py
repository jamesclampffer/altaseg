# Copyright 2026 Jim Clampffer. Created 2026-10-07.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Masker command line: serve the web UI."""

from __future__ import annotations

import argparse
import pathlib

import masker.execution.model
import masker.web
import masker.web.webcore


PORT = 5001


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the masker web UI.")
    parser.add_argument("--threshold", type=float,
                        default=masker.execution.model.MaskerDefaults.THRESHOLD,
                        help="detection threshold")
    parser.add_argument("--device", default=None, help="torch device (e.g. cpu)")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind")
    parser.add_argument("--port", type=int, default=PORT, help="port to bind")
    parser.add_argument(
        "--root", default=None, type=pathlib.Path,
        help="resolve relative image paths against this directory (default: cwd)",
    )
    args = parser.parse_args(argv)
    if args.root is not None and not args.root.is_dir():
        parser.error("--root is not a directory: {}".format(args.root))
    app = masker.web.create_app(
        device=args.device, threshold=args.threshold, root=args.root)
    masker.web.webcore.run_app(app, "masker", args.host, args.port)


if __name__ == "__main__":
    main()
