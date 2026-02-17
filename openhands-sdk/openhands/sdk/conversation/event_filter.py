"""Event filter for reducing context bloat by managing image content in chat history."""

from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.schema import Observation

from .event_filter_config import EventFilterConfig

logger = get_logger(__name__)


class EventFilter:
    """Filter and transform events to reduce context bloat.
    
    This class provides functionality to strip images from old observation events
    while keeping them in recent events, preventing exponential context growth
    in image-heavy workflows.
    
    Example:
        >>> from openhands.sdk.conversation.event_filter import EventFilter
        >>> from openhands.sdk.conversation.event_filter_config import EventFilterConfig
        >>> 
        >>> config = EventFilterConfig(recent_event_threshold=5)
        >>> filter = EventFilter(config)
        >>> 
        >>> # Filter a list of events
        >>> filtered_events = filter.filter_events(events)
    """
    
    def __init__(self, config: EventFilterConfig | None = None):
        """Initialize the event filter.
        
        Args:
            config: Configuration for filtering behavior. If None, uses default config.
        """
        self.config = config or EventFilterConfig()
    
    def should_strip_images(
        self,
        event: ObservationEvent,
        event_index: int,
        total_events: int,
        image_count: int = 0
    ) -> bool:
        """Determine if images should be stripped from this event.
        
        Args:
            event: The observation event to check
            event_index: Index of this event in the event list
            total_events: Total number of events in the list
            image_count: Number of images seen so far (for 'count' mode)
            
        Returns:
            True if images should be stripped from this event
        """
        if not self.config.enabled:
            return False
        
        # Don't apply to tools not in the target list
        if not self.config.should_apply_to_tool(event.tool_name):
            return False
        
        # Don't strip images from error observations (unless configured to do so)
        if event.observation.is_error and not self.config.strip_error_images:
            return False
        
        # Check if event has any images to strip
        has_images = any(
            isinstance(content, ImageContent)
            for content in event.observation.content
        )
        if not has_images:
            return False
        
        # Apply filtering based on mode
        if self.config.filter_mode == "age":
            # Age-based: strip if event is older than threshold
            events_from_end = total_events - event_index
            is_old = events_from_end > self.config.recent_event_threshold
            return is_old
        
        elif self.config.filter_mode == "count":
            # Count-based: strip if we've already kept max images
            return image_count >= self.config.max_images_to_keep
        
        return False
    
    def strip_images_from_observation(
        self,
        observation: Observation,
        event: ObservationEvent
    ) -> Observation:
        """Replace ImageContent with text descriptions.
        
        Args:
            observation: The observation to process
            event: The parent observation event (for context like tool_name)
            
        Returns:
            New observation with images replaced by text
        """
        new_content = []
        images_stripped = 0
        
        for content_item in observation.content:
            if isinstance(content_item, ImageContent):
                # Get context for replacement text
                path = getattr(observation, 'path', '')
                command = getattr(observation, 'command', 'view')
                
                # Replace with text description
                replacement_text = self.config.get_replacement_text(
                    path=path,
                    tool_name=event.tool_name,
                    command=command
                )
                new_content.append(TextContent(text=replacement_text))
                images_stripped += 1
            else:
                # Keep non-image content as-is
                new_content.append(content_item)
        
        if images_stripped > 0:
            logger.debug(
                f"Stripped {images_stripped} images from observation "
                f"(tool={event.tool_name}, id={event.id[:8]})"
            )
        
        # Create new observation with modified content
        # Use model_copy to preserve all other fields
        return observation.model_copy(update={"content": new_content})
    
    def filter_events(
        self,
        events: list[LLMConvertibleEvent]
    ) -> list[LLMConvertibleEvent]:
        """Filter events to reduce context bloat.
        
        Processes a list of events and strips images from older observation events
        while preserving images in recent events. The filtering behavior is controlled
        by the EventFilterConfig.
        
        Args:
            events: List of events to filter
            
        Returns:
            Filtered list of events with images selectively removed
        """
        if not self.config.enabled:
            return events
        
        filtered_events = []
        total = len(events)
        image_count = 0
        total_stripped = 0
        
        for idx, event in enumerate(events):
            if isinstance(event, ObservationEvent):
                # Check if we should strip images from this event
                should_strip = self.should_strip_images(
                    event, idx, total, image_count
                )
                
                if should_strip:
                    # Strip images and create new observation
                    new_observation = self.strip_images_from_observation(
                        event.observation,
                        event
                    )
                    
                    # Count how many images this event had
                    images_in_event = sum(
                        1 for c in event.observation.content
                        if isinstance(c, ImageContent)
                    )
                    total_stripped += images_in_event
                    
                    # Create new ObservationEvent with modified observation
                    new_event = event.model_copy(update={"observation": new_observation})
                    filtered_events.append(new_event)
                else:
                    # Keep event as-is (recent or doesn't match filter criteria)
                    # Count images to keep track for 'count' mode
                    images_in_event = sum(
                        1 for c in event.observation.content
                        if isinstance(c, ImageContent)
                    )
                    image_count += images_in_event
                    filtered_events.append(event)
            else:
                # Non-observation events pass through unchanged
                filtered_events.append(event)
        
        if total_stripped > 0:
            logger.info(
                f"Event filtering: Stripped {total_stripped} images from "
                f"{total} events, kept {image_count} recent images"
            )
        
        return filtered_events
