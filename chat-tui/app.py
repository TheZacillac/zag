"""
RAG Chat TUI - Interactive terminal interface for querying your RAG system

This is a Textual-based terminal user interface that provides an interactive chat
experience powered by the RAG system. It combines:
- Document retrieval via the /search endpoint (polars-worker)
- LLM generation via Ollama's streaming chat API
- A rich terminal UI with markdown rendering and clipboard support

Key features:
- Real-time streaming responses (shows text as it's generated)
- RAG context integration (queries document database for relevant chunks)
- Model selection (can switch between installed Ollama models)
- Copy/paste support for messages
- Keyboard shortcuts for navigation
- Status indicators for RAG and Ollama connectivity
"""

import os
import asyncio
import httpx
import time
import pyperclip  # Cross-platform clipboard library
import signal
import sys
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, Button, RichLog, Label, Select, LoadingIndicator
from textual.binding import Binding
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Configuration from environment variables
POLARS_API = os.getenv("POLARS_API", "http://polars-worker:8080")  # RAG query endpoint
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")  # Ollama API
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")  # Default LLM model

# Chat modes
MODE_RAG = "rag"        # Only answer from documents
MODE_HYBRID = "hybrid"  # Use documents when available, general knowledge otherwise
MODE_CHAT = "chat"      # Skip RAG, pure chat with LLM
DEFAULT_MODE = MODE_HYBRID


def format_duration(ms: int) -> str:
    """
    Format milliseconds into a human-readable duration string.

    Used in the status bar to show how long the last query took.
    Automatically scales to appropriate units (ms, s, m, h).

    Args:
        ms: Duration in milliseconds

    Returns:
        Formatted string like "123ms", "1.5s", "2m 30.0s", or "1h 15m 30.0s"
    """
    if ms < 1000:
        return f"{ms}ms"

    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


class StatusBar(Static):
    """
    Status bar showing real-time system health and query performance.

    Displays:
    - RAG service status (🟢 = healthy, 🔴 = down)
    - Ollama service status (🟢 = healthy, 🔴 = down)
    - Last query duration (how long the most recent query took)
    - Current model name (selected LLM model)

    Uses Textual's reactive properties to automatically update the display
    when any of these values change.
    """

    rag_status = reactive("🔴")  # Red circle until first successful RAG query
    ollama_status = reactive("🔴")  # Red circle until first successful Ollama call
    last_query_time = reactive("--")  # Shows duration like "1.2s" or "450ms"
    available_models = reactive([])  # List of models from Ollama API
    current_model = reactive(DEFAULT_CHAT_MODEL)  # Currently selected model

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_models = [DEFAULT_CHAT_MODEL]  # Start with default
        
    async def on_mount(self) -> None:
        """Load available models when status bar mounts"""
        await self.load_models()
        
    async def load_models(self) -> None:
        """Load available models from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{OLLAMA_HOST}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    if models:
                        self.available_models = models
        except Exception as e:
            self.available_models = [DEFAULT_CHAT_MODEL]  # Fallback to default

    def render(self):
        """Render the status bar with model selector"""
        return Text.from_markup(
            f"RAG: {self.rag_status} | Ollama: {self.ollama_status} | "
            f"Last Query: {self.last_query_time} | Model: {self.current_model}"
        )


class ThinkingPanel(Static):
    """
    Panel that displays the RAG system's retrieval and generation process.

    Shows step-by-step progress like:
    • Searching document database...
    • Found 5 relevant chunks
    • Generating response with context...

    This gives users visibility into what the system is doing, especially
    useful when queries take a few seconds to process.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = []  # List of thinking steps to display

    def add_step(self, step: str):
        """Add a new thinking step and refresh the display"""
        self.steps.append(step)
        self.update_display()

    def update_display(self):
        """Render all thinking steps in a styled panel"""
        content = "\n".join([f"• {step}" for step in self.steps])
        self.update(Panel(
            content,
            title="[bold #ca9ee6]🧠 Thinking Process[/bold #ca9ee6]",
            border_style="#ca9ee6",
            padding=(0, 1)
        ))

    def clear_steps(self):
        """Clear all steps (called when starting a new query)"""
        self.steps = []
        self.update("")


class RAGChatApp(App):
    """A Textual app for RAG-powered chat"""

    CSS = """
    Screen {
        background: #232634;
    }

    Header {
        background: #292c3c;
        color: #c6d0f5;
    }

    #header-title {
        color: #f2d5ce;
        text-style: bold;
    }


    #header-spacer {
        width: 1fr;
    }

    #status-bar {
        dock: top;
        height: 1;
        background: #292c3c;
        color: #c6d0f5;
        padding: 0 1;
    }

    .status-label {
        color: #838ba7;
    }

    .status-indicator {
        color: #c6d0f5;
    }

    .status-value {
        color: #a6d189;
    }

    .model-selector {
        background: #414559;
        color: #c6d0f5;
        border: solid #f2d5ce;
        margin: 0 1;
        width: 30;
        height: 3;
    }

    .model-selector:focus {
        background: #51576d;
        border: solid #a6d189;
    }

    #model-bar {
        dock: top;
        height: 3;
        background: #292c3c;
        color: #c6d0f5;
        padding: 0 1;
    }

    .model-label {
        color: #838ba7;
        margin-right: 1;
    }

    .model-spacer {
        width: 1fr;
    }

    #chat-container {
        height: 1fr;
        border: solid #babbf1;
        background: #232634;
        overflow-y: auto;
        padding: 1;
    }

    #thinking-panel {
        height: auto;
        margin: 1;
    }

    #loading-container {
        height: 3;
        background: #292c3c;
        padding: 0 1;
        display: none;
    }

    #loading-container.visible {
        display: block;
    }

    #loading-indicator {
        color: #ca9ee6;
    }

    .loading-text {
        color: #ca9ee6;
        margin-left: 2;
    }

    #input-container {
        height: auto;
        background: #292c3c;
        padding: 1;
    }

    #input {
        width: 1fr;
        margin-right: 1;
        background: #414559;
        color: #c6d0f5;
        border: solid #babbf1;
    }

    #send-btn {
        width: 12;
        background: #a6d189;
        color: #232634;
    }

    #send-btn:hover {
        background: #8bd5ca;
    }

    .system-msg {
        color: #838ba7;
        margin: 1;
        text-align: center;
    }

    .copyable {
        border: solid #f2d5ce;
        padding: 1;
        margin: 1;
    }

    Footer {
        background: #292c3c;
        color: #c6d0f5;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+y", "copy_last", "Copy Last Response"),
        Binding("ctrl+u", "copy_last_user", "Copy Last User Message"),
        Binding("ctrl+v", "paste", "Paste"),
        Binding("ctrl+a", "select_all", "Select All"),
        Binding("ctrl+t", "thumbs_up", "Thumbs Up (helpful)"),
        Binding("ctrl+d", "thumbs_down", "Thumbs Down (not helpful)"),
    ]

    def __init__(self):
        super().__init__()
        self.client = httpx.AsyncClient(timeout=180.0)
        self.last_response = ""
        self.last_user_message = ""
        self.query_start_time = 0
        self.chat_history = []  # Store chat messages for individual copying
        self.chat_model = DEFAULT_CHAT_MODEL  # Current model (instance variable)
        self.available_models = [DEFAULT_CHAT_MODEL]  # Available models from Ollama
        self.mode = DEFAULT_MODE  # Current chat mode (rag, hybrid, or chat)
        # Feedback tracking for progressive learning
        self.last_query = ""  # Last query text for feedback
        self.last_chunks = []  # Last retrieved chunks for feedback

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBar(id="status-bar")
        
        # Model selector bar (simplified - just show current model and mode)
        with Horizontal(id="model-bar"):
            yield Static(f"Model: {self.chat_model}", classes="model-label")
            yield Static(f"Mode: {self.mode}", id="mode-label", classes="model-label")
            yield Static("(Type '/model <name>' or '/mode <mode>' to change)", classes="model-hint")

        with Vertical():
            # Chat messages area
            yield RichLog(id="chat-container", highlight=True, markup=True)
            yield ThinkingPanel(id="thinking-panel")

            # Loading indicator (hidden by default)
            with Horizontal(id="loading-container"):
                yield LoadingIndicator(id="loading-indicator")
                yield Static("Processing...", classes="loading-text")

            # Input area
            with Horizontal(id="input-container"):
                yield Input(
                    placeholder="Ask a question... (Ctrl+Y: copy last, Ctrl+V: paste, Ctrl+A: select all)",
                    id="input"
                )
                yield Button("Send", variant="primary", id="send-btn")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app"""
        self.title = "RAG Chat"
        self.sub_title = f"🤖 {self.chat_model}"

        chat_log = self.query_one("#chat-container", RichLog)
        chat_log.write(
            Panel(
                "Welcome to RAG Chat! Ask questions about your documents.\n\n"
                f"🤖 Model: {self.chat_model}\n"
                f"🎯 Mode: {self.mode}\n"
                f"📚 RAG API: {POLARS_API}\n"
                f"🧠 Ollama: {OLLAMA_HOST}\n\n"
                "💡 Tips:\n"
                "  • Press Enter or click Send to submit\n"
                "  • Ctrl+Y to copy the last response\n"
                "  • Ctrl+U to copy the last user message\n"
                "  • Ctrl+V to paste from clipboard\n"
                "  • Ctrl+A to select all text in input\n"
                "  • Ctrl+L to clear chat\n"
                "  • Ctrl+C to quit\n"
                "  • Ctrl+T for thumbs up (helpful response)\n"
                "  • Ctrl+D for thumbs down (not helpful)\n"
                "  • /models to list available models\n"
                "  • /model <name> to change model\n"
                "  • /mode to see available modes\n"
                "  • /mode <mode> to change mode",
                title="[bold #f2d5ce]RAG Chat System[/bold #f2d5ce]",
                border_style="#f2d5ce"
            )
        )

        # Load available models from Ollama
        await self.load_models()

        # Check system status
        await self.check_system_status()
        


    async def load_models(self) -> None:
        """Load available models from Ollama API"""
        try:
            response = await self.client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                if models:
                    self.available_models = sorted(models)  # Sort alphabetically
        except Exception as e:
            # Fallback to default if Ollama is unavailable
            self.available_models = [self.chat_model]

    async def check_system_status(self):
        """Check if RAG and Ollama are accessible"""
        status_bar = self.query_one("#status-bar", StatusBar)

        # Check RAG
        try:
            resp = await self.client.get(f"{POLARS_API}/healthz", timeout=5.0)
            if resp.status_code == 200:
                status_bar.rag_status = "🟢"
        except Exception as e:
            status_bar.rag_status = "🔴"
            print(f"RAG health check failed: {e}")

        # Check Ollama
        try:
            resp = await self.client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                status_bar.ollama_status = "🟢"
        except Exception as e:
            status_bar.ollama_status = "🔴"
            print(f"Ollama health check failed: {e}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button click"""
        if event.button.id == "send-btn":
            await self.send_message()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input"""
        await self.send_message()

    async def send_message(self) -> None:
        """Send user message and get RAG response"""
        input_widget = self.query_one("#input", Input)
        chat_log = self.query_one("#chat-container", RichLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        user_msg = input_widget.value.strip()
        if not user_msg:
            return

        # Check for commands
        if user_msg.startswith("/model "):
            new_model = user_msg[7:].strip()  # Remove "/model " prefix
            if new_model:
                # Check if model is available
                if new_model in self.available_models:
                    self.chat_model = new_model
                    self.sub_title = f"🤖 {self.chat_model}"

                    # Update the model display
                    model_bar = self.query_one("#model-bar", Horizontal)
                    model_label = model_bar.query_one(".model-label", Static)
                    model_label.update(f"Model: {self.chat_model}")

                    # Notify user
                    chat_log = self.query_one("#chat-container", RichLog)
                    chat_log.write(f"[#a6d189]✓ Model changed to: {self.chat_model}[/#a6d189]")
                else:
                    # Model not found - show warning and available models
                    chat_log = self.query_one("#chat-container", RichLog)
                    chat_log.write(f"[#e5c890]⚠️ Model '{new_model}' not found in Ollama[/#e5c890]")
                    chat_log.write("[#8bd5ca]💡 Use '/models' to see available models[/#8bd5ca]")

                # Clear input and return
                input_widget.value = ""
                return
            else:
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write("[#e5c890]⚠️ Usage: /model <model_name>[/#e5c890]")
                chat_log.write("[#8bd5ca]💡 Use '/models' to see available models[/#8bd5ca]")
                input_widget.value = ""
                return
        elif user_msg == "/models":
            chat_log = self.query_one("#chat-container", RichLog)

            # Reload models from Ollama to get fresh list
            await self.load_models()

            if self.available_models:
                chat_log.write("[#8bd5ca]📋 Available models:[/#8bd5ca]")
                for model in self.available_models:
                    # Highlight current model
                    if model == self.chat_model:
                        chat_log.write(f"[#a6d189]  • {model} ← current[/#a6d189]")
                    else:
                        chat_log.write(f"[#c6d0f5]  • {model}[/#c6d0f5]")
                chat_log.write(f"[#8bd5ca]💡 Use '/model <name>' to change the current model[/#8bd5ca]")
            else:
                chat_log.write("[#e5c890]⚠️ No models available. Check Ollama connection.[/#e5c890]")

            input_widget.value = ""
            return
        elif user_msg.startswith("/mode"):
            chat_log = self.query_one("#chat-container", RichLog)

            # Check if user wants to change mode or just list modes
            parts = user_msg.split(maxsplit=1)
            if len(parts) == 1:
                # Just /mode - list available modes
                chat_log.write("[#8bd5ca]🎯 Available modes:[/#8bd5ca]")

                # RAG mode
                if self.mode == MODE_RAG:
                    chat_log.write(f"[#a6d189]  • {MODE_RAG} ← current[/#a6d189]")
                else:
                    chat_log.write(f"[#c6d0f5]  • {MODE_RAG}[/#c6d0f5]")
                chat_log.write("[#838ba7]    Only answer from RAG documents[/#838ba7]")

                # Hybrid mode
                if self.mode == MODE_HYBRID:
                    chat_log.write(f"[#a6d189]  • {MODE_HYBRID} ← current[/#a6d189]")
                else:
                    chat_log.write(f"[#c6d0f5]  • {MODE_HYBRID}[/#c6d0f5]")
                chat_log.write("[#838ba7]    Use documents when available, general knowledge otherwise[/#838ba7]")

                # Chat mode
                if self.mode == MODE_CHAT:
                    chat_log.write(f"[#a6d189]  • {MODE_CHAT} ← current[/#a6d189]")
                else:
                    chat_log.write(f"[#c6d0f5]  • {MODE_CHAT}[/#c6d0f5]")
                chat_log.write("[#838ba7]    Skip RAG, pure chat with LLM[/#838ba7]")

                chat_log.write(f"[#8bd5ca]💡 Use '/mode <mode>' to change the current mode[/#8bd5ca]")
            else:
                # /mode <mode_name> - change mode
                new_mode = parts[1].strip().lower()
                if new_mode in [MODE_RAG, MODE_HYBRID, MODE_CHAT]:
                    self.mode = new_mode

                    # Update the mode display in UI
                    mode_label = self.query_one("#mode-label", Static)
                    mode_label.update(f"Mode: {self.mode}")

                    # Show success message with description
                    mode_descriptions = {
                        MODE_RAG: "Only answer from RAG documents",
                        MODE_HYBRID: "Use documents when available, general knowledge otherwise",
                        MODE_CHAT: "Skip RAG, pure chat with LLM"
                    }
                    chat_log.write(f"[#a6d189]✓ Mode changed to: {self.mode}[/#a6d189]")
                    chat_log.write(f"[#838ba7]  {mode_descriptions[self.mode]}[/#838ba7]")
                else:
                    chat_log.write(f"[#e5c890]⚠️ Invalid mode: {new_mode}[/#e5c890]")
                    chat_log.write(f"[#8bd5ca]💡 Valid modes: {MODE_RAG}, {MODE_HYBRID}, {MODE_CHAT}[/#8bd5ca]")

            input_widget.value = ""
            return

        # Store user message for copying and feedback
        self.last_user_message = user_msg
        self.last_query = user_msg  # Store for feedback
        self.last_chunks = []  # Reset chunks for new query

        # Clear input
        input_widget.value = ""
        self.query_start_time = time.time()

        # Show user message
        chat_log.write(
            Panel(
                user_msg,
                title="[bold #8bd5ca]You[/bold #8bd5ca] [dim #838ba7](Ctrl+U to copy)[/dim #838ba7]",
                border_style="#8bd5ca"
            )
        )

        # Show loading indicator
        loading_container = self.query_one("#loading-container", Horizontal)
        loading_container.add_class("visible")

        # Create thinking panel
        thinking_panel = self.query_one("#thinking-panel", ThinkingPanel)
        thinking_panel.clear_steps()

        def add_thinking_step(step: str):
            thinking_panel.add_step(step)

        try:
            # Initialize context and timing variables
            context_chunks = []
            context = None
            rag_time = 0

            # ===== PHASE 1: RETRIEVAL (RAG Query) =====
            # Query the polars-worker API to find relevant document chunks
            # Skip RAG in CHAT mode

            if self.mode == MODE_CHAT:
                # CHAT mode: Skip RAG entirely
                add_thinking_step("💬 Chat mode - skipping document search")
            else:
                # RAG or HYBRID mode: Query the document database
                add_thinking_step("📝 Embedding your question...")

                add_thinking_step("🔍 Searching document database...")
                rag_start = time.time()

                # Call the /search endpoint which:
                # 1. Embeds the user's question
                # 2. Searches pgvector for similar chunks
                # 3. Returns top 5 most relevant chunks
                rag_response = await self.client.get(
                    f"{POLARS_API}/search",
                    params={"q": user_msg, "k": 5}
                )
                rag_response.raise_for_status()
                rag_data = rag_response.json()
                rag_time = int((time.time() - rag_start) * 1000)

                context_chunks = rag_data.get("chunks", [])
                self.last_chunks = context_chunks  # Store for feedback

                # ===== PHASE 2: CONTEXT FORMATTING =====
                # Format retrieved chunks into context string for LLM

                if context_chunks:
                    add_thinking_step(f"✅ Found {len(context_chunks)} relevant chunks ({format_duration(rag_time)})")

                    # Show chunk details with similarity scores
                    chunk_details = []
                    for i, chunk in enumerate(context_chunks, 1):
                        dist = chunk.get('distance', 0)
                        # Convert cosine distance to similarity percentage
                        # Distance ranges from 0 (identical) to 2 (opposite)
                        similarity = (1 - dist) * 100
                        chunk_details.append(
                            f"{i}. {chunk['title']} (similarity: {similarity:.1f}%)\n"
                            f"   Preview: {chunk['text'][:100]}..."
                        )

                    chat_log.write(Panel(
                        "\n\n".join(chunk_details),
                        title=f"[bold #e5c890]📚 Retrieved Context ({len(context_chunks)} chunks)[/bold #e5c890]",
                        border_style="#e5c890"
                    ))

                    # Build context string for LLM prompt
                    context = "\n\n".join([
                        f"[Document: {chunk['title']}]\n{chunk['text']}"
                        for chunk in context_chunks
                    ])
                else:
                    add_thinking_step("⚠️ No relevant context found")
                    context = None

            # ===== PHASE 3: LLM GENERATION (Streaming) =====
            # Send context + question to Ollama and stream the response

            add_thinking_step(f"💭 Generating response with {self.chat_model}...")

            # System prompt and message construction varies by mode
            if self.mode == MODE_CHAT:
                # CHAT mode: Pure conversation, no document context
                system_prompt = (
                    "You are a helpful AI assistant. Answer the user's questions using your "
                    "general knowledge and reasoning abilities."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ]
            elif self.mode == MODE_RAG:
                # RAG mode: Strict document-only responses
                if context:
                    system_prompt = (
                        "You are a helpful assistant that answers questions based strictly on the provided context. "
                        "Use only the context to answer the user's question. If the context doesn't contain "
                        "relevant information, say so honestly and do not use outside knowledge."
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_msg}"}
                    ]
                else:
                    system_prompt = (
                        "You are a helpful assistant. The document database did not find any relevant information "
                        "for this question. Politely inform the user that no relevant documents were found."
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {user_msg}"}
                    ]
            else:  # MODE_HYBRID
                # HYBRID mode: Use documents when available, general knowledge otherwise
                if context:
                    system_prompt = (
                        "You are a helpful assistant that answers questions based on the provided context. "
                        "Use the context to answer the user's question. If the context doesn't fully answer "
                        "the question, you may supplement with your general knowledge, but prioritize the context."
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_msg}"}
                    ]
                else:
                    system_prompt = (
                        "You are a helpful AI assistant. No relevant documents were found in the database, "
                        "so answer the user's question using your general knowledge and reasoning abilities."
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {user_msg}"}
                    ]

            # Stream response from Ollama for real-time display
            response_parts = []  # Use list for efficient concatenation
            llm_start = time.time()

            # Use streaming mode to show text as it's generated
            async with self.client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/chat",
                json={"model": self.chat_model, "messages": messages, "stream": True}
            ) as stream:
                # Show context summary
                if context_chunks:
                    chat_log.write(Panel(
                        f"✓ Using {len(context_chunks)} relevant chunks from database",
                        title="[bold #e5c890]📚 Context[/bold #e5c890]",
                        border_style="#e5c890"
                    ))

                # Stream tokens from Ollama's response
                # Each line is a JSON object with a "message" field containing new tokens
                import json
                async for line in stream.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                response_parts.append(chunk["message"]["content"])
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines

            # Join all response parts into final text
            response_text = "".join(response_parts)

            # ===== PHASE 4: DISPLAY FINAL RESPONSE =====
            # Calculate timing and show complete response

            llm_time = int((time.time() - llm_start) * 1000)
            total_time = int((time.time() - self.query_start_time) * 1000)

            # Update status bar with total time
            status_bar.last_query_time = format_duration(total_time)

            # Store for Ctrl+Y copy action
            self.last_response = response_text

            # Display final response with timing breakdown
            chat_log.write(Panel(
                Markdown(response_text),  # Render markdown formatting
                title=f"[bold #a6d189]🤖 Assistant[/bold #a6d189] (generated in {format_duration(llm_time)}) [dim #838ba7](Ctrl+Y to copy)[/dim #838ba7]",
                border_style="#a6d189",
                subtitle=f"[dim #838ba7]Total: {format_duration(total_time)} | RAG: {format_duration(rag_time)} | LLM: {format_duration(llm_time)}[/dim #838ba7]"
            ))

        except httpx.HTTPError as e:
            # HTTP errors (API down, network issues, etc.)
            chat_log.write(f"[#e78284]❌ HTTP Error: {e}[/#e78284]")
            status_bar.last_query_time = "ERROR"
        except Exception as e:
            # Unexpected errors (parsing issues, etc.)
            chat_log.write(f"[#e78284]❌ Unexpected error: {e}[/#e78284]")
            status_bar.last_query_time = "ERROR"
        finally:
            # Hide loading indicator when done (success or error)
            loading_container = self.query_one("#loading-container", Horizontal)
            loading_container.remove_class("visible")
            thinking_panel.clear_steps()

    def action_clear(self) -> None:
        """Clear chat history"""
        chat_log = self.query_one("#chat-container", RichLog)
        chat_log.clear()
        chat_log.write(
            Panel(
                "Chat cleared. Ask a new question!",
                title="[bold #f2d5ce]RAG Chat[/bold #f2d5ce]",
                border_style="#f2d5ce"
            )
        )

    def action_copy_last(self) -> None:
        """Copy last response to clipboard"""
        if self.last_response:
            try:
                pyperclip.copy(self.last_response)
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write("[#a6d189]✓ Last response copied to clipboard![/#a6d189]")
            except Exception as e:
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write(f"[#e78284]❌ Failed to copy: {e}[/#e78284]")
        else:
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write("[#e5c890]⚠️ No response to copy yet[/#e5c890]")

    def action_paste(self) -> None:
        """Paste from clipboard into input field"""
        try:
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                input_widget = self.query_one("#input", Input)
                # Insert clipboard content at cursor position
                current_value = input_widget.value
                cursor_position = input_widget.cursor_position
                new_value = current_value[:cursor_position] + clipboard_text + current_value[cursor_position:]
                input_widget.value = new_value
                input_widget.cursor_position = cursor_position + len(clipboard_text)
                
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write(f"[#a6d189]✓ Text pasted from clipboard! ({len(clipboard_text)} chars)[/#a6d189]")
            else:
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write("[#e5c890]⚠️ Clipboard is empty[/#e5c890]")
        except pyperclip.PyperclipException as e:
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write(f"[#e78284]❌ Clipboard not available in Docker container[/#e78284]")
            chat_log.write("[#8bd5ca]💡 Tip: Type your text directly in the input field[/#8bd5ca]")
        except Exception as e:
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write(f"[#e78284]❌ Failed to paste: {e}[/#e78284]")

    def action_select_all(self) -> None:
        """Select all text in the input field"""
        input_widget = self.query_one("#input", Input)
        if input_widget.value:
            input_widget.cursor_position = 0
            input_widget.selection = (0, len(input_widget.value))
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write("[#a6d189]✓ All text selected![/#a6d189]")

    def action_copy_last_user(self) -> None:
        """Copy last user message to clipboard"""
        if self.last_user_message:
            self.copy_message(self.last_user_message, "user message")
        else:
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write("[#e5c890]⚠️ No user message to copy yet[/#e5c890]")

    def action_thumbs_up(self) -> None:
        """Mark last response as helpful (thumbs up)"""
        self.run_worker(self.send_feedback(was_helpful=True))

    def action_thumbs_down(self) -> None:
        """Mark last response as not helpful (thumbs down)"""
        self.run_worker(self.send_feedback(was_helpful=False))

    async def send_feedback(self, was_helpful: bool) -> None:
        """
        Send feedback for all chunks in the last response.

        Args:
            was_helpful: True for thumbs up, False for thumbs down
        """
        chat_log = self.query_one("#chat-container", RichLog)

        if not self.last_query:
            chat_log.write("[#e5c890]⚠️ No query to provide feedback for yet[/#e5c890]")
            return

        if not self.last_chunks:
            chat_log.write("[#e5c890]⚠️ No chunks were retrieved for this query[/#e5c890]")
            return

        try:
            # Send feedback for each chunk that was returned
            feedback_count = 0
            for chunk in self.last_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id:
                    response = await self.client.post(
                        f"{POLARS_API}/feedback",
                        json={
                            "query": self.last_query,
                            "chunk_id": chunk_id,
                            "was_helpful": was_helpful,
                            "clicked": True  # Assume they saw it if they're rating it
                        }
                    )
                    if response.status_code == 200:
                        feedback_count += 1

            # Show success message
            emoji = "👍" if was_helpful else "👎"
            helpful_text = "helpful" if was_helpful else "not helpful"
            chat_log.write(
                f"[#a6d189]✓ {emoji} Feedback recorded: {feedback_count} chunk(s) marked as {helpful_text}[/#a6d189]"
            )
            chat_log.write(
                "[#8bd5ca]💡 Your feedback helps improve future search results![/#8bd5ca]"
            )

        except httpx.HTTPError as e:
            chat_log.write(f"[#e78284]❌ Failed to send feedback: {e}[/#e78284]")
        except Exception as e:
            chat_log.write(f"[#e78284]❌ Unexpected error: {e}[/#e78284]")

    def copy_message(self, message_text: str, message_type: str = "message") -> None:
        """Copy a specific message to clipboard"""
        try:
            # Clean up the message text (remove markdown formatting for plain text)
            import re
            clean_text = re.sub(r'\[/?[^\]]*\]', '', message_text)  # Remove all [tag] and [/tag] patterns
            clean_text = clean_text.strip()
            
            if not clean_text:
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write(f"[#e5c890]⚠️ {message_type.title()} is empty, nothing to copy[/#e5c890]")
                return
            
            pyperclip.copy(clean_text)
            chat_log = self.query_one("#chat-container", RichLog)
            # Show a brief success message
            chat_log.write(f"[#a6d189]✓ {message_type.title()} copied to clipboard! ({len(clean_text)} chars)[/#a6d189]")
        except pyperclip.PyperclipException as e:
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write(f"[#e78284]❌ Clipboard not available in Docker container[/#e78284]")
            chat_log.write(f"[#e5c890]📋 Text to copy ({len(clean_text)} chars):[/#e5c890]")
            chat_log.write(f"[dim #838ba7]{clean_text}[/dim #838ba7]")
            chat_log.write("[#8bd5ca]💡 Tip: Select and copy the text above manually[/#8bd5ca]")
        except Exception as e:
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write(f"[#e78284]❌ Failed to copy {message_type}: {e}[/#e78284]")

    async def on_unmount(self) -> None:
        """Clean up on exit"""
        if hasattr(self, 'client') and self.client:
            await self.client.aclose()
        # Restore terminal settings
        self.restore_terminal()

    def restore_terminal(self):
        """Restore terminal to normal state"""
        try:
            # Reset terminal attributes
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                # Disable mouse reporting
                sys.stdout.write('\x1b[?1000l')  # Disable mouse tracking
                sys.stdout.write('\x1b[?1002l')  # Disable mouse drag
                sys.stdout.write('\x1b[?1003l')  # Disable mouse move
                sys.stdout.write('\x1b[?1006l')  # Disable SGR mouse mode
                sys.stdout.write('\x1b[?1007l')  # Disable mouse wheel
                # Reset cursor
                sys.stdout.write('\x1b[?25h')    # Show cursor
                sys.stdout.write('\x1b[0m')      # Reset all attributes
                sys.stdout.flush()
        except Exception:
            pass  # Ignore errors during cleanup


def signal_handler(signum, frame):
    """Handle signals to ensure clean exit"""
    print("\n\nCleaning up terminal...")
    try:
        # Restore terminal settings
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            sys.stdout.write('\x1b[?1000l')  # Disable mouse tracking
            sys.stdout.write('\x1b[?1002l')  # Disable mouse drag
            sys.stdout.write('\x1b[?1003l')  # Disable mouse move
            sys.stdout.write('\x1b[?1006l')  # Disable SGR mouse mode
            sys.stdout.write('\x1b[?1007l')  # Disable mouse wheel
            sys.stdout.write('\x1b[?25h')    # Show cursor
            sys.stdout.write('\x1b[0m')      # Reset all attributes
            sys.stdout.flush()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        app = RAGChatApp()
        app.run()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"Error: {e}")
        signal_handler(signal.SIGTERM, None)
    finally:
        # Final cleanup
        signal_handler(signal.SIGTERM, None)
