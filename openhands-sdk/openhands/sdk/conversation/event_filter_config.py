"""Event filter configuration for managing context bloat."""

from dataclasses import dataclass, field


@dataclass
class EventFilterConfig:
    """Configuration for event filtering to reduce context bloat.
    
    This configuration is particularly useful for pipelines that process many images,
    where keeping all image content in the chat history would lead to:
    - Exponential context growth
    - Degraded model performance
    - Higher API costs
    - Slower response times
    
    Example:
        >>> config = EventFilterConfig(
        ...     enabled=True,
        ...     recent_event_threshold=5,
        ...     target_tools=["file_editor"]
        ... )
        >>> # Use this config when creating a conversation
        >>> conversation = LocalConversation(
        ...     agent_config=agent_config,
        ...     event_filter_config=config
        ... )
    """
    
    # Enable/disable image filtering globally
    enabled: bool = True
    
    # Number of recent observation events to keep images for
    # Example: If set to 5, only the last 5 image observations will retain their images
    recent_event_threshold: int = 5
    
    # Specific tools to apply filtering to
    # None means apply to all tools with image content
    target_tools: list[str] | None = field(default_factory=lambda: ["file_editor"])
    
    # Template for replacement text when stripping images
    # Available placeholders: {path}, {tool_name}, {command}
    replacement_template: str = "[Previously viewed image: {path}]"
    
    # Whether to strip images from error observations
    # Usually you want to keep error context, so default is False
    strip_error_images: bool = False
    
    # Filter mode: 'age' (based on event age) or 'count' (keep only N most recent)
    filter_mode: str = "age"  # Options: "age", "count"
    
    # For 'count' mode: maximum number of images to keep in total
    max_images_to_keep: int = 10
    
    def should_apply_to_tool(self, tool_name: str) -> bool:
        """Check if filtering should apply to a specific tool.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if filtering should be applied to this tool
        """
        if not self.enabled:
            return False
        
        if self.target_tools is None:
            # Apply to all tools
            return True
        
        return tool_name in self.target_tools
    
    def get_replacement_text(self, path: str = "", tool_name: str = "", command: str = "") -> str:
        """Generate replacement text for stripped images.
        
        Args:
            path: File path of the image
            tool_name: Name of the tool that generated the observation
            command: Command that was executed
            
        Returns:
            Formatted replacement text
        """
        return self.replacement_template.format(
            path=path or "unknown",
            tool_name=tool_name or "unknown",
            command=command or "view"
        )
