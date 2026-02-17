"""Tests for EventFilter functionality."""

import pytest
from unittest.mock import Mock

from openhands.sdk.conversation.event_filter import EventFilter
from openhands.sdk.conversation.event_filter_config import EventFilterConfig
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool.schema import Observation


# Mock observation class for testing
class MockFileEditorObservation(Observation):
    """Mock file editor observation for testing."""
    command: str = "view"
    path: str = "/path/to/image.png"


def create_mock_image_observation(path: str = "/test/image.png") -> ObservationEvent:
    """Create a mock observation event with image content."""
    observation = MockFileEditorObservation(
        content=[
            TextContent(text=f"Image file {path} read successfully."),
            ImageContent(image_urls=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="]),
        ],
        path=path,
    )
    
    return ObservationEvent(
        source="environment",
        tool_name="file_editor",
        tool_call_id="test_call_123",
        observation=observation,
        action_id="action_123",
    )


def create_mock_text_observation() -> ObservationEvent:
    """Create a mock observation event with only text content."""
    observation = MockFileEditorObservation(
        content=[TextContent(text="Some text result")],
        path="/test/file.txt",
    )
    
    return ObservationEvent(
        source="environment",
        tool_name="file_editor",
        tool_call_id="test_call_124",
        observation=observation,
        action_id="action_124",
    )


class TestEventFilterConfig:
    """Tests for EventFilterConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EventFilterConfig()
        
        assert config.enabled is True
        assert config.recent_event_threshold == 5
        assert config.target_tools == ["file_editor"]
        assert "[Previously viewed image:" in config.replacement_template
        assert config.strip_error_images is False
        assert config.filter_mode == "age"
    
    def test_should_apply_to_tool(self):
        """Test tool filtering logic."""
        config = EventFilterConfig(target_tools=["file_editor", "custom_tool"])
        
        assert config.should_apply_to_tool("file_editor") is True
        assert config.should_apply_to_tool("custom_tool") is True
        assert config.should_apply_to_tool("other_tool") is False
    
    def test_should_apply_to_all_tools_when_none(self):
        """Test that None target_tools applies to all."""
        config = EventFilterConfig(target_tools=None)
        
        assert config.should_apply_to_tool("any_tool") is True
        assert config.should_apply_to_tool("file_editor") is True
    
    def test_get_replacement_text(self):
        """Test replacement text generation."""
        config = EventFilterConfig(
            replacement_template="[Viewed {path} with {tool_name}]"
        )
        
        text = config.get_replacement_text(
            path="/test/image.png",
            tool_name="file_editor",
            command="view"
        )
        
        assert text == "[Viewed /test/image.png with file_editor]"


class TestEventFilter:
    """Tests for EventFilter."""
    
    def test_disabled_filter_returns_unchanged(self):
        """Test that disabled filter returns events unchanged."""
        config = EventFilterConfig(enabled=False)
        filter = EventFilter(config)
        
        events = [
            create_mock_image_observation("/image1.png"),
            create_mock_image_observation("/image2.png"),
        ]
        
        filtered = filter.filter_events(events)
        
        assert len(filtered) == 2
        assert filtered == events
        # Check images are still present
        assert any(
            isinstance(c, ImageContent)
            for c in filtered[0].observation.content
        )
    
    def test_age_based_filtering(self):
        """Test age-based filtering keeps recent images."""
        config = EventFilterConfig(
            enabled=True,
            recent_event_threshold=2,
            filter_mode="age"
        )
        filter = EventFilter(config)
        
        # Create 5 image observations
        events = [
            create_mock_image_observation(f"/image{i}.png")
            for i in range(5)
        ]
        
        filtered = filter.filter_events(events)
        
        assert len(filtered) == 5
        
        # First 3 events should have images stripped (old)
        for i in range(3):
            has_images = any(
                isinstance(c, ImageContent)
                for c in filtered[i].observation.content
            )
            assert not has_images, f"Event {i} should have images stripped"
            
            # Should have replacement text
            has_replacement = any(
                isinstance(c, TextContent) and "Previously viewed" in c.text
                for c in filtered[i].observation.content
            )
            assert has_replacement, f"Event {i} should have replacement text"
        
        # Last 2 events should keep images (recent)
        for i in range(3, 5):
            has_images = any(
                isinstance(c, ImageContent)
                for c in filtered[i].observation.content
            )
            assert has_images, f"Event {i} should keep images"
    
    def test_count_based_filtering(self):
        """Test count-based filtering keeps only N images."""
        config = EventFilterConfig(
            enabled=True,
            filter_mode="count",
            max_images_to_keep=2
        )
        filter = EventFilter(config)
        
        # Create 5 image observations
        events = [
            create_mock_image_observation(f"/image{i}.png")
            for i in range(5)
        ]
        
        filtered = filter.filter_events(events)
        
        # Count total images kept
        total_images = sum(
            1 for event in filtered
            for c in event.observation.content
            if isinstance(c, ImageContent)
        )
        
        assert total_images == 2, "Should keep exactly 2 images"
    
    def test_non_file_editor_tools_pass_through(self):
        """Test that non-target tools are not filtered."""
        config = EventFilterConfig(
            enabled=True,
            target_tools=["file_editor"],
            recent_event_threshold=0  # Would strip all if applied
        )
        filter = EventFilter(config)
        
        # Create observation from different tool
        obs = MockFileEditorObservation(
            content=[
                TextContent(text="Custom tool result"),
                ImageContent(image_urls=["data:image/png;base64,ABC123"]),
            ],
        )
        event = ObservationEvent(
            source="environment",
            tool_name="custom_tool",
            tool_call_id="test_call",
            observation=obs,
            action_id="action_123",
        )
        
        filtered = filter.filter_events([event])
        
        # Image should be kept since tool is not targeted
        has_images = any(
            isinstance(c, ImageContent)
            for c in filtered[0].observation.content
        )
        assert has_images
    
    def test_text_only_observations_unchanged(self):
        """Test that observations without images pass through."""
        config = EventFilterConfig(enabled=True)
        filter = EventFilter(config)
        
        events = [create_mock_text_observation()]
        filtered = filter.filter_events(events)
        
        assert len(filtered) == 1
        assert filtered[0] == events[0]
    
    def test_mixed_events(self):
        """Test filtering with mixed event types."""
        config = EventFilterConfig(
            enabled=True,
            recent_event_threshold=1
        )
        filter = EventFilter(config)
        
        events = [
            create_mock_image_observation("/old_image.png"),
            create_mock_text_observation(),
            create_mock_image_observation("/new_image.png"),
        ]
        
        filtered = filter.filter_events(events)
        
        assert len(filtered) == 3
        
        # First image should be stripped
        assert not any(
            isinstance(c, ImageContent)
            for c in filtered[0].observation.content
        )
        
        # Last image should be kept
        assert any(
            isinstance(c, ImageContent)
            for c in filtered[2].observation.content
        )
    
    def test_error_observations_not_stripped_by_default(self):
        """Test that error observations keep images by default."""
        config = EventFilterConfig(
            enabled=True,
            recent_event_threshold=0,  # Would strip all
            strip_error_images=False
        )
        filter = EventFilter(config)
        
        # Create error observation with image
        obs = MockFileEditorObservation(
            content=[
                TextContent(text="Error occurred"),
                ImageContent(image_urls=["data:image/png;base64,ABC123"]),
            ],
            is_error=True,
        )
        event = ObservationEvent(
            source="environment",
            tool_name="file_editor",
            tool_call_id="test_call",
            observation=obs,
            action_id="action_123",
        )
        
        filtered = filter.filter_events([event])
        
        # Image should be kept in error observation
        has_images = any(
            isinstance(c, ImageContent)
            for c in filtered[0].observation.content
        )
        assert has_images
    
    def test_strip_images_from_observation(self):
        """Test the image stripping method directly."""
        config = EventFilterConfig()
        filter = EventFilter(config)
        
        event = create_mock_image_observation("/test.png")
        original_obs = event.observation
        
        new_obs = filter.strip_images_from_observation(original_obs, event)
        
        # Check images are removed
        has_images = any(
            isinstance(c, ImageContent)
            for c in new_obs.content
        )
        assert not has_images
        
        # Check replacement text is added
        has_replacement = any(
            isinstance(c, TextContent) and "Previously viewed" in c.text
            for c in new_obs.content
        )
        assert has_replacement
        
        # Check other fields are preserved
        assert new_obs.path == original_obs.path
        assert new_obs.command == original_obs.command


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
