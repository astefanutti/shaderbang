# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

import argparse
import re


class Metadata(argparse.Action):

    def __call__(self, _, namespace, values, option=None):
        m = re.search(r"^(\w+)\.(\w+)$", values[0])
        if not m:
            raise ValueError(f"value {values[0]} for option {option} must match <UNIFORM>.KEY")
        metadata = getattr(namespace, self.dest)
        if not m.group(1) in metadata:
            metadata[m.group(1)] = {}
        metadata[m.group(1)] = {**metadata[m.group(1)], **{m.group(2): values[1]}}
