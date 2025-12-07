"""
Multi-turn Conversation Memory for RoboBrain 2.0

This module provides conversation memory that maintains context across
multiple queries about the same scene or across different images.
"""

import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class Turn:
    """A single conversation turn (user query + model response)."""
    role: str  # "user" or "assistant"
    content: str
    image_path: Optional[str] = None
    task: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ConversationMemory:
    """
    Maintains multi-turn conversation history with image context.
    
    Features:
    - Tracks conversation turns with timestamps
    - Associates images with conversation context
    - Supports saving/loading conversations
    - Builds context prompts for the model
    """
    
    turns: List[Turn] = field(default_factory=list)
    current_image: Optional[str] = None
    max_turns: int = 20  # Maximum turns to keep in memory
    
    def add_user_turn(self, content: str, image_path: Optional[str] = None, task: str = "general") -> None:
        """Add a user query to the conversation."""
        # Update current image if a new one is provided
        if image_path:
            self.current_image = image_path
            
        turn = Turn(
            role="user",
            content=content,
            image_path=image_path or self.current_image,
            task=task
        )
        self.turns.append(turn)
        self._trim_history()
    
    def add_assistant_turn(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add model response to the conversation."""
        turn = Turn(
            role="assistant",
            content=content,
            image_path=self.current_image,
            metadata=metadata or {}
        )
        self.turns.append(turn)
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Keep only the most recent turns to prevent context overflow."""
        if len(self.turns) > self.max_turns:
            # Keep the most recent turns
            self.turns = self.turns[-self.max_turns:]
    
    def get_context_prompt(self, include_images: bool = True) -> str:
        """
        Build a context string from conversation history.
        
        This creates a summary that can be prepended to new queries
        to give the model awareness of previous exchanges.
        """
        if not self.turns:
            return ""
        
        context_parts = ["[Previous conversation context]"]
        
        for turn in self.turns[-10:]:  # Last 10 turns for context
            role_label = "User" if turn.role == "user" else "Assistant"
            
            if turn.image_path and include_images and turn.role == "user":
                context_parts.append(f"{role_label} (with image): {turn.content}")
            else:
                context_parts.append(f"{role_label}: {turn.content}")
        
        context_parts.append("[Current query]")
        return "\n".join(context_parts)
    
    def get_conversation_summary(self) -> str:
        """Generate a brief summary of the conversation for the model."""
        if len(self.turns) < 2:
            return ""
        
        # Extract key information discussed
        topics = []
        objects_mentioned = set()
        
        for turn in self.turns:
            # Simple extraction of nouns/objects (could be enhanced with NLP)
            content_lower = turn.content.lower()
            for word in ["cup", "table", "person", "chair", "banana", "remote", 
                        "cat", "dog", "car", "door", "window", "bottle"]:
                if word in content_lower:
                    objects_mentioned.add(word)
        
        summary = f"We've been discussing an image"
        if objects_mentioned:
            summary += f" containing: {', '.join(objects_mentioned)}"
        summary += f". ({len(self.turns)} exchanges so far)"
        
        return summary
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns = []
        self.current_image = None
    
    def save(self, filepath: str) -> None:
        """Save conversation to JSON file."""
        data = {
            "current_image": self.current_image,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "image_path": t.image_path,
                    "task": t.task,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata
                }
                for t in self.turns
            ]
        }
        
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Conversation saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load conversation from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.current_image = data.get("current_image")
        self.turns = [
            Turn(
                role=t["role"],
                content=t["content"],
                image_path=t.get("image_path"),
                task=t.get("task"),
                timestamp=t.get("timestamp", ""),
                metadata=t.get("metadata", {})
            )
            for t in data.get("turns", [])
        ]
        print(f"Loaded {len(self.turns)} turns from {filepath}")
    
    def __len__(self) -> int:
        return len(self.turns)
    
    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self.turns)}, current_image={self.current_image})"


class MultiTurnInference:
    """
    Wrapper around RoboBrain model that maintains conversation memory.
    
    Usage:
        chat = MultiTurnInference(model)
        chat.set_image("scene.jpg")
        response1 = chat.ask("What objects are on the table?")
        response2 = chat.ask("Which one should I grab first?")  # Remembers context
    """
    
    def __init__(self, model, repo_dir=None):
        """
        Initialize with a loaded RoboBrain model.
        
        Args:
            model: UnifiedInference model instance
            repo_dir: Path to RoboBrain repo (for demo assets)
        """
        self.model = model
        self.repo_dir = repo_dir
        self.memory = ConversationMemory()
        self.use_context = True  # Whether to include conversation history
    
    def set_image(self, image_path: str) -> None:
        """Set the current image for conversation."""
        self.memory.current_image = image_path
        print(f"📷 Image set: {image_path}")
    
    def ask(
        self, 
        prompt: str, 
        image_path: Optional[str] = None,
        task: str = "general",
        enable_thinking: bool = True,
        do_sample: bool = True,
        plot: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question with conversation context.
        
        Args:
            prompt: The user's question
            image_path: Optional new image (uses current if not provided)
            task: Task type (general, grounding, affordance, trajectory, pointing)
            enable_thinking: Whether to enable model's thinking mode
            do_sample: Whether to use sampling
            plot: Whether to plot results on image (for visual tasks)
            
        Returns:
            Dict with 'answer', optionally 'thinking', 'output_image', and 'context_used'
        """
        # Use provided image or fall back to current
        img = image_path or self.memory.current_image
        
        if not img:
            return {"error": "No image set. Use set_image() or provide image_path."}
        
        # Build context-aware prompt
        if self.use_context and len(self.memory) > 0:
            context = self.memory.get_context_prompt()
            full_prompt = f"{context}\n{prompt}"
        else:
            full_prompt = prompt
        
        # Add user turn to memory
        self.memory.add_user_turn(prompt, image_path, task)
        
        # Run inference
        try:
            result = self.model.inference(
                full_prompt, 
                img, 
                task=task,
                enable_thinking=enable_thinking,
                do_sample=do_sample,
                plot=plot
            )
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            # Remove the failed user turn to keep history clean
            if self.memory.turns and self.memory.turns[-1].role == "user":
                self.memory.turns.pop()
            return {"error": error_msg, "answer": error_msg, "turn_number": len(self.memory), "context_used": False}
        
        # Handle different result formats
        if result is None:
            answer = "No response from model"
            thinking = ""
            output_image = None
        elif isinstance(result, dict):
            answer = result.get("answer", str(result))
            thinking = result.get("thinking", "")
            output_image = result.get("output_image")
        elif isinstance(result, str):
            answer = result
            thinking = ""
            output_image = None
        else:
            answer = str(result)
            thinking = ""
            output_image = None
        
        # Add assistant response to memory
        self.memory.add_assistant_turn(answer, {"thinking": thinking, "task": task, "output_image": output_image})
        
        # Return enriched result
        return {
            "answer": answer,
            "thinking": thinking,
            "output_image": output_image,
            "context_used": self.use_context and len(self.memory) > 2,
            "turn_number": len(self.memory),
            "task": task
        }
    
    def ground(self, object_description: str, image_path: Optional[str] = None, plot: bool = False) -> Dict[str, Any]:
        """Shortcut for grounding task."""
        return self.ask(object_description, image_path, task="grounding", plot=plot)
    
    def get_affordance(self, action: str, image_path: Optional[str] = None, plot: bool = False) -> Dict[str, Any]:
        """Shortcut for affordance task."""
        return self.ask(action, image_path, task="affordance", plot=plot)
    
    def get_trajectory(self, action: str, image_path: Optional[str] = None, plot: bool = False) -> Dict[str, Any]:
        """Shortcut for trajectory task."""
        return self.ask(action, image_path, task="trajectory", plot=plot)
    
    def point_at(self, description: str, image_path: Optional[str] = None, plot: bool = False) -> Dict[str, Any]:
        """Shortcut for pointing task."""
        return self.ask(description, image_path, task="pointing", plot=plot)
    
    def reset(self) -> None:
        """Clear conversation memory and start fresh."""
        self.memory.clear()
        print("🔄 Conversation memory cleared.")
    
    def save_conversation(self, filepath: str) -> None:
        """Save current conversation to file."""
        self.memory.save(filepath)
    
    def load_conversation(self, filepath: str) -> None:
        """Load a previous conversation."""
        self.memory.load(filepath)
    
    def show_history(self) -> None:
        """Print conversation history."""
        print("\n" + "="*50)
        print("📜 CONVERSATION HISTORY")
        print("="*50)
        
        for i, turn in enumerate(self.memory.turns):
            role_icon = "👤" if turn.role == "user" else "🤖"
            print(f"\n{role_icon} [{turn.role.upper()}] (Turn {i+1})")
            if turn.image_path and turn.role == "user":
                print(f"   📷 Image: {turn.image_path}")
            if turn.task and turn.role == "user":
                print(f"   📋 Task: {turn.task}")
            print(f"   {turn.content[:200]}{'...' if len(turn.content) > 200 else ''}")
        
        print("\n" + "="*50)
