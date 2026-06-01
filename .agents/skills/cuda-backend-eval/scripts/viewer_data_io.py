#!/usr/bin/env python3
"""Read and write benchmark-viewer JSON, including sharded collections."""

from __future__ import annotations

from viewer_data_io_impl.naming import capture_import_name
from viewer_data_io_impl.naming import record_filename
from viewer_data_io_impl.naming import record_identity
from viewer_data_io_impl.naming import record_prefix
from viewer_data_io_impl.naming import slug
from viewer_data_io_impl.read import expand_manifest
from viewer_data_io_impl.read import is_sharded_manifest
from viewer_data_io_impl.read import load_json
from viewer_data_io_impl.read import load_sharded_collection
from viewer_data_io_impl.read import manifest_record_files
from viewer_data_io_impl.sidecars import expand_record_sidecars
from viewer_data_io_impl.sidecars import load_sidecar_list
from viewer_data_io_impl.sidecars import split_record_sidecars
from viewer_data_io_impl.sidecars import write_sidecar_list
from viewer_data_io_impl.write import collection_key
from viewer_data_io_impl.write import sharded_target
from viewer_data_io_impl.write import write_json
from viewer_data_io_impl.write import write_sharded_collection
