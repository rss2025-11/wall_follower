#!/usr/bin/env python3
"""
ROS2 Bag Analysis Tool

This script opens a ROS2 bag file and provides tools for analyzing and visualizing the data.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple

# ROS2 bag reading
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class BagReader:
    """Class for reading ROS2 bag files."""

    def __init__(self, bag_path: str):
        """
        Initialize the BagReader.

        Args:
            bag_path: Path to the ROS2 bag file
        """
        self.bag_path = Path(bag_path)

        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

        # Initialize ROS2
        rclpy.init()

        # Set up storage and converter options
        self.storage_options = StorageOptions(
            uri=str(self.bag_path), storage_id="sqlite3"
        )
        self.converter_options = ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )

        # Metadata
        self._topic_types = None
        self._start_time = None
        self._end_time = None
        self._message_counts = None

    def get_topics_and_types(self) -> Dict[str, str]:
        """
        Get all topics and their types from the bag file.

        Returns:
            Dictionary mapping topic names to their message types
        """
        if self._topic_types is not None:
            return self._topic_types

        reader = SequentialReader()
        reader.open(self.storage_options, self.converter_options)

        topic_types = {}
        for topic_metadata in reader.get_all_topics_and_types():
            topic_types[topic_metadata.name] = topic_metadata.type

        self._topic_types = topic_types
        return topic_types

    def get_message_count(self, topic: Optional[str] = None) -> Dict[str, int]:
        """
        Get the number of messages for each topic or a specific topic.

        Args:
            topic: Topic name (if None, get counts for all topics)

        Returns:
            Dictionary mapping topic names to message counts, or a single count if topic is specified
        """
        if self._message_counts is not None:
            if topic is not None:
                return self._message_counts.get(topic, 0)
            return self._message_counts

        reader = SequentialReader()
        reader.open(self.storage_options, self.converter_options)

        topic_types = self.get_topics_and_types()
        message_counts = {topic: 0 for topic in topic_types}

        while reader.has_next():
            topic_name, _, _ = reader.read_next()
            if topic_name in message_counts:
                message_counts[topic_name] += 1

        self._message_counts = message_counts

        if topic is not None:
            return message_counts.get(topic, 0)
        return message_counts

    def get_bag_duration(self) -> float:
        """
        Get the duration of the bag file in seconds.

        Returns:
            Duration in seconds
        """
        if self._start_time is not None and self._end_time is not None:
            return self._end_time - self._start_time

        reader = SequentialReader()
        reader.open(self.storage_options, self.converter_options)

        start_time = None
        end_time = None

        while reader.has_next():
            _, _, timestamp = reader.read_next()
            timestamp_sec = timestamp * 1e-9  # Convert to seconds

            if start_time is None or timestamp_sec < start_time:
                start_time = timestamp_sec
            if end_time is None or timestamp_sec > end_time:
                end_time = timestamp_sec

        self._start_time = start_time
        self._end_time = end_time

        return end_time - start_time

    def read_messages(
        self, topics: Optional[List[str]] = None
    ) -> Iterator[Tuple[str, Any, float]]:
        """
        Read messages from the bag file.

        Args:
            topics: List of topics to read from (if None, read from all topics)

        Yields:
            Tuples of (topic_name, message, timestamp)
        """
        reader = SequentialReader()
        reader.open(self.storage_options, self.converter_options)

        topic_types = self.get_topics_and_types()

        # Filter topics if specified
        if topics:
            selected_topics = {
                topic: topic_types[topic] for topic in topics if topic in topic_types
            }
            if len(selected_topics) < len(topics):
                missing = set(topics) - set(selected_topics.keys())
                print(f"Warning: Topics not found in bag: {missing}")
        else:
            selected_topics = topic_types

        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()

            # Skip topics we're not interested in
            if topic_name not in selected_topics:
                continue

            # Get message type
            msg_type = get_message(selected_topics[topic_name])

            # Deserialize message
            msg = deserialize_message(data, msg_type)

            # Convert timestamp to seconds
            timestamp_sec = timestamp * 1e-9

            yield topic_name, msg, timestamp_sec

    def get_messages_by_topic(
        self, topics: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """
        Get all messages organized by topic.

        Args:
            topics: List of topics to read from (if None, read from all topics)

        Returns:
            Dictionary mapping topic names to lists of (message, timestamp) tuples
        """
        messages_by_topic = {}

        for topic_name, msg, timestamp in self.read_messages(topics):
            if topic_name not in messages_by_topic:
                messages_by_topic[topic_name] = []

            messages_by_topic[topic_name].append((msg, timestamp))

        return messages_by_topic

    def get_message_field_values(
        self, topic: str, field_path: str
    ) -> List[Tuple[Any, float]]:
        """
        Get values for a specific field from all messages on a topic.

        Args:
            topic: Topic name
            field_path: Dot-separated path to the field (e.g., "pose.pose.position.x")

        Returns:
            List of (value, timestamp) tuples
        """
        values = []

        for topic_name, msg, timestamp in self.read_messages([topic]):
            try:
                value = self._get_message_field(msg, field_path)
                values.append((value, timestamp))
            except AttributeError as e:
                print(f"Warning: {e}")

        return values

    def _get_message_field(self, msg: Any, field_path: str) -> Any:
        """
        Get a field from a ROS message using a dot-separated path.

        Args:
            msg: ROS message
            field_path: Dot-separated path to the field (e.g., "pose.pose.position.x")

        Returns:
            Field value
        """
        if not field_path:
            return msg

        fields = field_path.split(".")
        value = msg

        for field in fields:
            if hasattr(value, field):
                value = getattr(value, field)
            else:
                raise AttributeError(f"Field '{field}' not found in message")

        return value

    def get_message_fields(self, topic: str, max_depth: int = 3) -> List[str]:
        """
        Get all available fields for a topic by inspecting the first message.

        Args:
            topic: Topic name
            max_depth: Maximum depth to explore in nested messages

        Returns:
            List of field paths
        """
        # Get the first message for this topic
        for topic_name, msg, _ in self.read_messages([topic]):
            # Recursively get all fields
            fields = []

            def explore_fields(obj, prefix="", depth=0):
                if depth >= max_depth:
                    return

                # Get all attributes that don't start with underscore
                for attr in dir(obj):
                    if attr.startswith("_"):
                        continue

                    # Skip methods
                    if callable(getattr(obj, attr)):
                        continue

                    value = getattr(obj, attr)

                    # Add the field
                    field_path = f"{prefix}.{attr}" if prefix else attr
                    fields.append(field_path)

                    # Explore nested objects
                    if hasattr(value, "__dict__") or hasattr(value, "__slots__"):
                        explore_fields(value, field_path, depth + 1)

            explore_fields(msg)
            return fields

        return []

    def close(self):
        """Clean up resources."""
        rclpy.shutdown()


# def main():
#     """Main function to run the bag analyzer."""
#     parser = argparse.ArgumentParser(description="Analyze ROS2 bag files")
#     parser.add_argument("bag_path", help="Path to the ROS2 bag file")
#     parser.add_argument("--topics", nargs="+", help="Topics to analyze")

#     args = parser.parse_args()

#     try:
#         reader = BagReader(args.bag_path)

#         # Print bag information
#         print(f"Analyzing bag file: {args.bag_path}")

#         # Get topics and types
#         topic_types = reader.get_topics_and_types()
#         print("\nAvailable topics:")
#         for topic, msg_type in topic_types.items():
#             print(f"  - {topic} ({msg_type})")

#         # Get message counts
#         message_counts = reader.get_message_count()
#         print("\nMessage counts:")
#         for topic, count in message_counts.items():
#             print(f"  - {topic}: {count} messages")

#         # Get bag duration
#         duration = reader.get_bag_duration()
#         print(f"\nBag duration: {duration:.2f} seconds")

#         # If topics are specified, print some information about them
#         if args.topics:
#             for topic in args.topics:
#                 if topic in topic_types:
#                     print(f"\nTopic: {topic}")

#                     # Get fields for this topic
#                     fields = reader.get_message_fields(topic)
#                     print("  Available fields (first few):")
#                     for field in fields[:5]:  # Show first 5 fields
#                         print(f"    - {field}")

#                     # Get message count for this topic
#                     count = reader.get_message_count(topic)
#                     print(f"  Message count: {count}")
#                 else:
#                     print(f"\nTopic not found: {topic}")

#     finally:
#         reader.close()


# if __name__ == "__main__":
#     main()
