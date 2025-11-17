"""LMDB writer for GraphNeT's data conversion pipeline.

Saves each event as a key/value pair where the key is the event index
(`event_no`) and the value is a user-serialized blob. Optionally, a
`DataRepresentation` can be injected to persist pre-built representations
instead of raw tables.
"""

import os
import lmdb
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .graphnet_writer import GraphNeTWriter
from graphnet.models.data_representation import DataRepresentation
from graphnet.data.utilities.lmdb_utilities import (
    get_serialization_method_name,
    _get_data_representation_metadata_dict,
)


def _serialize_pickle(obj: Any) -> bytes:
    """Serialize object using pickle."""
    import pickle  # type: ignore

    return pickle.dumps(obj)


def _serialize_json(obj: Any) -> bytes:
    """Serialize object using json."""
    import json  # type: ignore

    return json.dumps(obj).encode("utf-8")


def _serialize_msgpack(obj: Any) -> bytes:
    """Serialize object using msgpack."""
    try:
        import msgpack  # type: ignore
    except ImportError as e:
        raise ImportError("msgpack is not installed.") from e

    return msgpack.packb(obj, use_bin_type=True)


def _serialize_dill(obj: Any) -> bytes:
    """Serialize object using dill."""
    try:
        import dill  # type: ignore
    except ImportError as e:
        raise ImportError("dill is not installed.") from e

    return dill.dumps(obj)


class LMDBWriter(GraphNeTWriter):
    """Writer that exports events to an LMDB database.

    Each event is stored under key = bytes(str(event_no)) and value =
    bytes produced by a user-selected serialization function.
    """

    def __init__(
        self,
        index_column: str = "event_no",
        map_size_bytes: int = 8 * 1024 * 1024 * 1024,
        serialization: Union[str, Callable[[Any], bytes]] = "pickle",
        data_representation: Optional[
            Union[DataRepresentation, List[DataRepresentation]]
        ] = None,
        pulsemap_extractor_name: Optional[str] = None,
        truth_extractor_name: Optional[str] = None,
        truth_label_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `LMDBWriter`.

        Args:
            index_column: Column used as the per-event key
                (default: `event_no`).
            map_size_bytes: LMDB map size. Defaults to 8 GiB.
            serialization: Either a string in {"pickle", "json", "msgpack",
                "dill"}, or a callable that takes an object and returns bytes.
            data_representation: Optional `DataRepresentation` instance or list
                of instances. If provided together with extractor names and
                truth labels, the stored value will contain a
                "data_representations" field with outputs from each
                `data_representation.forward(...)` keyed by class name.
            pulsemap_extractor_name: Name of the extractor providing
                pulse-level features.
            truth_extractor_name: Name of the extractor providing event-level
                truth labels.
            truth_label_names: Names of truth columns to include.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        self._file_extension = ".lmdb"
        # Receive per-event lists from the DataConverter
        self._merge_dataframes = False
        self._index_column = index_column
        self._map_size_bytes = map_size_bytes

        # Store the serialization method name for metadata
        if isinstance(serialization, str):
            self._serialization_method = serialization
        else:
            self._serialization_method = "__custom__"

        self._serializer = self._resolve_serializer(serialization)

        # Convert single DataRepresentation to list for consistent handling
        if data_representation is None:
            self._data_representations: Optional[List[DataRepresentation]] = (
                None
            )
        elif isinstance(data_representation, list):
            self._data_representations = data_representation
        else:
            self._data_representations = [data_representation]

        self._pulsemap_name = pulsemap_extractor_name
        self._truth_name = truth_extractor_name
        self._truth_label_names = truth_label_names

    def _resolve_serializer(
        self, serialization: Union[str, Callable[[Any], bytes]]
    ) -> Callable[[Any], bytes]:
        if callable(serialization):
            return serialization
        if serialization == "pickle":
            return _serialize_pickle
        if serialization == "json":
            return _serialize_json
        if serialization == "msgpack":
            return _serialize_msgpack
        if serialization == "dill":
            return _serialize_dill
        raise ValueError(
            "Unsupported serialization. Use 'pickle', 'json', "
            "'msgpack', 'dill', or a callable."
        )

    def _store_serialization_metadata(self, txn: lmdb.Transaction) -> None:
        """Store serialization method metadata in the LMDB database.

        This metadata allows future readers to determine which
        serialization method was used to create the database.
        """
        metadata_key = b"__meta_serialization__"
        metadata_value = self._serialization_method.encode("utf-8")
        txn.put(metadata_key, metadata_value, overwrite=True)

    def _get_data_representation_field_names(
        self,
    ) -> Dict[str, DataRepresentation]:
        """Get mapping of field names to data representations.

        Returns a dictionary where keys are the field names used to
        store data representations (matching the keys in
        data_representations_output) and values are the corresponding
        DataRepresentation instances.
        """
        if self._data_representations is None:
            return {}

        field_name_to_rep: Dict[str, DataRepresentation] = {}
        class_name_counts: Dict[str, int] = {}

        for data_rep in self._data_representations:
            base_class_name = data_rep.__class__.__name__

            # Track occurrences of each class name
            if base_class_name not in class_name_counts:
                # First occurrence - check if base name is available
                if base_class_name in field_name_to_rep:
                    # Base name already taken (shouldn't happen), use _0
                    class_name_counts[base_class_name] = 0
                    key_name = f"{base_class_name}_0"
                else:
                    # First occurrence - use base name
                    class_name_counts[base_class_name] = 0
                    key_name = base_class_name
            else:
                # We've seen this class name before - increment and append tag
                class_name_counts[base_class_name] += 1
                count = class_name_counts[base_class_name]
                key_name = f"{base_class_name}_{count}"

            field_name_to_rep[key_name] = data_rep

        return field_name_to_rep

    def _store_data_representation_metadata(
        self, txn: lmdb.Transaction
    ) -> None:
        """Store data representation metadata in the LMDB database.

        This metadata allows future readers to determine which data
        representations were used and their configurations.
        """
        if self._data_representations is None:
            return

        # Get mapping of field names to data representations
        field_name_to_rep = self._get_data_representation_field_names()

        # Build metadata dictionary with configs
        metadata_dict: Dict[str, Any] = {}
        for field_name, data_rep in field_name_to_rep.items():
            # Get the config and convert to dict
            config = data_rep.config
            # Use dict() to get a serializable representation
            metadata_dict[field_name] = config

        # Serialize the metadata dictionary
        metadata_key = b"__meta_data_representations__"
        metadata_value = self._serializer(metadata_dict)
        txn.put(metadata_key, metadata_value, overwrite=True)

    def _event_dict_from_merged_tables(
        self, tables: Dict[str, pd.DataFrame], event_no: int
    ) -> Dict[str, pd.DataFrame]:
        """Slice all tables to a single event and return a per-event dict."""
        event_tables: Dict[str, pd.DataFrame] = {}
        for name, df in tables.items():
            if len(df) == 0:
                continue
            if (
                self._index_column not in df.columns
                and df.index.name != self._index_column
            ):
                # If a table wasn't indexed properly, skip it.
                continue
            if self._index_column in df.columns:
                mask = df[self._index_column] == event_no
                sliced = df.loc[mask]
            else:
                # Indexed by event_no already
                try:
                    sliced = df.loc[event_no]
                    if isinstance(sliced, pd.Series):
                        sliced = sliced.to_frame().T
                except KeyError:
                    continue
            if len(sliced) > 0:
                event_tables[name] = sliced.reset_index(drop=True)
        return event_tables

    def _build_value_from_tables(
        self, per_event_tables: Dict[str, pd.DataFrame]
    ) -> Any:
        """Build the object to serialize for a single event from tables."""
        extractor_dict = {
            name: df.to_dict(orient="list")
            for name, df in per_event_tables.items()
        }
        rep_dict = {}
        if self._data_representations is not None:
            if self._pulsemap_name is None or self._truth_name is None:
                raise ValueError(
                    "pulsemap_extractor_name and truth_extractor_name must"
                    "be set when using data_representation."
                )
            pulse_df = per_event_tables.get(
                self._pulsemap_name, pd.DataFrame()
            )
            truth_df = per_event_tables.get(self._truth_name, pd.DataFrame())
            if pulse_df.empty or truth_df.empty:
                return {
                    name: df.to_dict(orient="list")
                    for name, df in per_event_tables.items()
                }

            # Prepare truth data
            truth_row = truth_df.iloc[0].to_dict()
            if self._truth_label_names is not None:
                truth_row = {
                    k: truth_row[k]
                    for k in self._truth_label_names
                    if k in truth_row
                }
            truth_dicts = [truth_row]

            # Process each data representation
            data_representations_output: Dict[str, Any] = {}
            class_name_counts: Dict[str, int] = {}

            for data_rep in self._data_representations:
                # Get feature names for this representation
                feature_names = getattr(
                    data_rep, "_input_feature_names", None
                ) or getattr(data_rep, "input_feature_names", None)
                if feature_names is None:
                    feature_names = [
                        c for c in pulse_df.columns if c != self._index_column
                    ]

                x = pulse_df[feature_names].to_numpy()

                # Get the output from this data representation
                rep_output = data_rep.forward(
                    input_features=x,
                    input_feature_names=list(feature_names),
                    truth_dicts=truth_dicts,
                )  # type: ignore[arg-type]

                # Generate unique key for this representation
                base_class_name = data_rep.__class__.__name__

                # Track occurrences of each class name
                if base_class_name not in class_name_counts:
                    # First occurrence - check if base name is available
                    if base_class_name in data_representations_output:
                        # Base name already taken (shouldn't happen), use _0
                        class_name_counts[base_class_name] = 0
                        key_name = f"{base_class_name}_0"
                    else:
                        # First occurrence - use base name
                        class_name_counts[base_class_name] = 0
                        key_name = base_class_name
                else:
                    # We've seen this class name before -
                    # increment and append tag
                    class_name_counts[base_class_name] += 1
                    # Break for max line length..
                    count = class_name_counts[base_class_name]
                    key_name = f"{base_class_name}_{count}"

                data_representations_output[key_name] = rep_output

            rep_dict = {"data_representations": data_representations_output}

        if len(rep_dict) > 0:
            extractor_dict.update(rep_dict)
        return extractor_dict

    def _save_file(
        self,
        data: Union[Dict[str, pd.DataFrame], Dict[str, List[pd.DataFrame]]],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save data to an LMDB database.

        Each input file becomes its own LMDB environment located at
        `output_file_path` (a directory with `.lmdb` suffix).
        """
        # LMDB expects a directory path. Create a directory for this file.
        env_path = output_file_path
        os.makedirs(env_path, exist_ok=True)

        # Identify unique event ids or infer from list lengths
        if len(data) == 0 or n_events == 0:
            self.warning(
                f"No data to write for {output_file_path}. Skipping.."
            )
            return
        env = lmdb.open(
            env_path,
            map_size=self._map_size_bytes,
            subdir=True,
            lock=True,
            max_dbs=1,
        )
        written = 0
        with env.begin(write=True) as txn:
            # Store serialization method metadata
            self._store_serialization_metadata(txn)
            # Store data representation metadata
            self._store_data_representation_metadata(txn)

            is_merged = all(isinstance(v, pd.DataFrame) for v in data.values())
            if is_merged:
                written += self._save_from_merged(txn, data)
            else:
                written += self._save_from_lists(txn, data)

        env.sync()
        env.close()
        self.debug(f"Wrote {written} events to {env_path}")

    def _save_from_merged(
        self,
        txn: lmdb.Transaction,
        data: Dict[str, pd.DataFrame],
    ) -> int:
        """Write events when input is a dict of merged DataFrames."""
        from typing import cast

        merged_tables = cast(Dict[str, pd.DataFrame], data)
        ref_name = min(
            merged_tables.keys(), key=lambda k: len(merged_tables[k])
        )
        ref_df = merged_tables[ref_name]
        if self._index_column in ref_df.columns:
            event_ids = pd.unique(ref_df[self._index_column])
        else:
            event_ids = pd.unique(ref_df.index)
        written = 0
        for event_no in event_ids:
            try:
                event_int = int(event_no)
            except Exception:
                continue
            per_event = self._event_dict_from_merged_tables(
                merged_tables, event_int
            )
            if len(per_event) == 0:
                continue
            value_obj = self._build_value_from_tables(per_event)
            try:
                value_bytes = self._serializer(value_obj)
            except Exception as e:
                self.error(f"Serialization failed for event {event_int}: {e}")
                continue
            key_bytes = str(event_int).encode("utf-8")
            txn.put(key_bytes, value_bytes, overwrite=True)
            written += 1
        return written

    def _save_from_lists(
        self,
        txn: lmdb.Transaction,
        data: Dict[str, List[pd.DataFrame]],
    ) -> int:
        """Write events when input is a dict of lists of DataFrames."""
        from typing import cast

        lists_dict = cast(Dict[str, List[pd.DataFrame]], data)
        event_count = max(len(v) for v in lists_dict.values())
        written = 0
        for event_idx in range(event_count):
            per_event_for_lists: Dict[str, pd.DataFrame] = {}
            event_no_val: Optional[int] = None
            for name, lst in lists_dict.items():
                if event_idx >= len(lst):
                    continue
                df_event = lst[event_idx]
                if (
                    self._index_column in df_event.columns
                    and len(df_event[self._index_column]) > 0
                    and event_no_val is None
                ):
                    try:
                        event_no_val = int(
                            df_event[self._index_column].iloc[0]
                        )
                    except Exception:
                        pass
                per_event_for_lists[name] = df_event.reset_index(drop=True)
            if len(per_event_for_lists) == 0 or event_no_val is None:
                continue
            value_obj = self._build_value_from_tables(per_event_for_lists)
            try:
                value_bytes = self._serializer(value_obj)
            except Exception as e:
                self.error(
                    f"Serialization failed for event {event_no_val}: {e}"
                )
                continue
            key_bytes = str(event_no_val).encode("utf-8")
            txn.put(key_bytes, value_bytes, overwrite=True)
            written += 1
        return written

    def merge_files(
        self,
        files: List[str],
        output_dir: str,
        target_name: str = "merged.lmdb",
        allow_overwrite: bool = False,
    ) -> None:
        """Merge multiple LMDB environments into one.

        Note: Keys are assumed unique across inputs. If a duplicate key is
        encountered and `allow_overwrite=False`, the key is skipped.
        """
        if len(files) == 0:
            self.warning("No LMDB files provided for merging. Skipping.")
            return

        # Calculate total size of source files to set appropriate map_size
        total_size = 0
        for src in files:
            src_env_path = src
            if os.path.isdir(src_env_path):
                # Sum up all file sizes in the LMDB directory
                for root, dirs, filenames in os.walk(src_env_path):
                    for filename in filenames:
                        filepath = os.path.join(root, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
            elif os.path.exists(src_env_path):
                total_size += os.path.getsize(src_env_path)

        # Set map_size to total size plus 20% overhead,
        # or use default if calculated size is smaller
        map_size = max(
            self._map_size_bytes,
            int(total_size * 1.2),  # 20% overhead for LMDB metadata
        )

        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, target_name)
        os.makedirs(target_path, exist_ok=True)

        target_env = lmdb.open(
            target_path,
            map_size=map_size,
            subdir=True,
            lock=True,
            max_dbs=1,
        )
        with target_env.begin(write=True) as target_txn:
            # Determine serialization method from first source file
            # and store it in the merged database
            serialization_method = None
            for src in files:
                src_env_path = src if os.path.isdir(src) else src
                try:
                    serialization_method = get_serialization_method_name(
                        src_env_path
                    )
                    if serialization_method is not None:
                        break
                except Exception:
                    continue

            # If no metadata found, use the current writer's method
            if serialization_method is None:
                serialization_method = self._serialization_method

            # Store metadata in merged database
            metadata_key = b"__meta_serialization__"
            metadata_value = serialization_method.encode("utf-8")
            target_txn.put(metadata_key, metadata_value, overwrite=True)

            # Determine data representation metadata from first source file
            # and store it in the merged database
            data_rep_metadata = None
            for src in files:
                src_env_path = src if os.path.isdir(src) else src
                try:
                    data_rep_metadata = _get_data_representation_metadata_dict(
                        src_env_path
                    )
                    if data_rep_metadata is not None:
                        break
                except Exception:
                    continue

            # If metadata found, store it in merged database
            # Use the serializer that matches the merged database's
            # serialization method
            if data_rep_metadata is not None:
                try:
                    merged_serializer = self._resolve_serializer(
                        serialization_method
                    )
                except ValueError:
                    # Fall back to writer's serializer for custom methods
                    merged_serializer = self._serializer
                metadata_key_dr = b"__meta_data_representations__"
                metadata_value_dr = merged_serializer(data_rep_metadata)
                target_txn.put(
                    metadata_key_dr, metadata_value_dr, overwrite=True
                )

            for src in files:
                if not os.path.isdir(src):
                    # Expecting LMDB dir; if a file path was passed, try dir
                    src_env_path = src
                else:
                    src_env_path = src
                try:
                    src_env = lmdb.open(
                        src_env_path, readonly=True, lock=False, subdir=True
                    )
                except Exception:
                    # Skip non-lmdb inputs
                    continue
                with src_env.begin(write=False) as src_txn:
                    cursor = src_txn.cursor()
                    for key, val in cursor:
                        # Skip metadata keys from source (we've already set it)
                        if (
                            key == b"__meta_serialization__"
                            or key == b"__meta_data_representations__"
                        ):
                            continue
                        try:
                            if (not allow_overwrite) and target_txn.get(
                                key
                            ) is not None:
                                continue
                        except Exception as e:
                            self.error(f"Error checking for overwrite: {e}")
                            continue
                        try:
                            target_txn.put(key, val, overwrite=True)
                        except Exception as e:
                            self.error(f"Error writing key {key}: {e}")
                            continue
                src_env.close()

        target_env.sync()
        target_env.close()
        self.info(f"Merged {len(files)} LMDB databases into {target_path}")
