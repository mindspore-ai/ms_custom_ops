# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""gen op list"""

import os
import glob
from pathlib import Path

def main():
    # Define the base directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_path, "../")

    # Find all markdown files in the ops directory and its subdirectories
    md_files = glob.glob(os.path.join(base_dir, "ops", "**", "*.md"), recursive=True)

    # Prepare output markdown content
    output_lines = ["# ms_custom_ops算子列表\n\n"]

    # Process each markdown file
    for md_file in md_files:
        # Get the relative path from base directory
        file_path = Path(md_file)

        # Get the filename without extension
        op_name = file_path.stem

        if op_name.endswith('_doc'):
            # Find the last occurrence of 'doc' and remove it along with the preceding underscore if present
            idx = op_name.rfind('_doc')
            op_name = op_name[:idx]

        # Create a markdown link to the file
        # Use the relative path for the link
        link_path = os.path.relpath(md_file, base_dir + "/docs")
        output_lines.append(f"[{op_name}]({link_path})\n")

    output_lines.sort()                # Sort them alphabetically

    index = 1
    output_lines_with_prefix = output_lines[:1]
    for line in output_lines[1:]:
        output_lines_with_prefix.append(f"{index}. {line}")

    # Write the output to a markdown file
    output_file = os.path.join(base_dir, "docs/op_list.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(output_lines_with_prefix))

if __name__ == "__main__":
    main()
