"""
RAG Chat TUI - Interactive terminal interface for querying your RAG system
"""

import os
import asyncio
import httpx
import time
import pyperclip
import signal
import sys
import termios
import tty
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, Button, RichLog, Label, Select
from textual.binding import Binding
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Configuration from environment
POLARS_API = os.getenv("POLARS_API", "http://polars-worker:8080")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")


def format_duration(ms: int) -> str:
    """Format milliseconds into a human-readable duration string"""
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
    """Status bar showing system status with model selector"""

    rag_status = reactive("🔴")
    ollama_status = reactive("🔴")
    last_query_time = reactive("--")
    available_models = reactive([])
    current_model = reactive(CHAT_MODEL)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_models = [CHAT_MODEL]  # Start with default
        
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
            self.available_models = [CHAT_MODEL]  # Fallback to default

    def render(self):
        """Render the status bar with model selector"""
        return Text.from_markup(
            f"RAG: {self.rag_status} | Ollama: {self.ollama_status} | "
            f"Last Query: {self.last_query_time} | Model: {self.current_model}"
        )


class ThinkingPanel(Static):
    """Displays the model's thinking process"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = []

    def add_step(self, step: str):
        """Add a thinking step"""
        self.steps.append(step)
        self.update_display()

    def update_display(self):
        """Update the display with current steps"""
        content = "\n".join([f"• {step}" for step in self.steps])
        self.update(Panel(
            content,
            title="[bold #ca9ee6]🧠 Thinking Process[/bold #ca9ee6]",
            border_style="#ca9ee6",
            padding=(0, 1)
        ))

    def clear_steps(self):
        """Clear all steps"""
        self.steps = []


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
    ]

    def __init__(self):
        super().__init__()
        self.client = httpx.AsyncClient(timeout=180.0)
        self.last_response = ""
        self.last_user_message = ""
        self.query_start_time = 0
        self.chat_history = []  # Store chat messages for individual copying

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBar(id="status-bar")
        
        # Model selector bar (simplified - just show current model)
        with Horizontal(id="model-bar"):
            yield Static(f"Model: {CHAT_MODEL}", classes="model-label")
            yield Static("(Type '/model <name>' to change)", classes="model-hint")

        with Vertical():
            # Chat messages area
            yield RichLog(id="chat-container", highlight=True, markup=True)

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
        self.sub_title = f"🤖 {CHAT_MODEL}"

        chat_log = self.query_one("#chat-container", RichLog)
        chat_log.write(
            Panel(
                "Welcome to RAG Chat! Ask questions about your documents.\n\n"
                f"🤖 Model: {CHAT_MODEL}\n"
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
                "  • /models to list available models\n"
                "  • /model <name> to change model",
                title="[bold #f2d5ce]RAG Chat System[/bold #f2d5ce]",
                border_style="#f2d5ce"
            )
        )

        # Check system status
        await self.check_system_status()
        


    async def check_system_status(self):
        """Check if RAG and Ollama are accessible"""
        status_bar = self.query_one("#status-bar", StatusBar)

        # Check RAG
        try:
            resp = await self.client.get(f"{POLARS_API}/healthz", timeout=5.0)
            if resp.status_code == 200:
                status_bar.rag_status = "🟢"
        except:
            status_bar.rag_status = "🔴"

        # Check Ollama
        try:
            resp = await self.client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                status_bar.ollama_status = "🟢"
        except:
            status_bar.ollama_status = "🔴"

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
                global CHAT_MODEL
                CHAT_MODEL = new_model
                self.sub_title = f"🤖 {CHAT_MODEL}"
                
                # Update the model display
                model_bar = self.query_one("#model-bar", Horizontal)
                model_label = model_bar.query_one(".model-label", Static)
                model_label.update(f"Model: {CHAT_MODEL}")
                
                # Notify user
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write(f"[#a6d189]✓ Model changed to: {CHAT_MODEL}[/#a6d189]")
                
                # Clear input and return
                input_widget.value = ""
                return
            else:
                chat_log = self.query_one("#chat-container", RichLog)
                chat_log.write("[#e5c890]⚠️ Usage: /model <model_name>[/#e5c890]")
                chat_log.write("[#8bd5ca]💡 Available models: gpt-oss:120b, all-minilm:latest, embeddinggemma:latest[/#8bd5ca]")
                input_widget.value = ""
                return
        elif user_msg == "/models":
            chat_log = self.query_one("#chat-container", RichLog)
            chat_log.write("[#8bd5ca]📋 Available models:[/#8bd5ca]")
            chat_log.write("[#c6d0f5]  • gpt-oss:120b (default)[/#c6d0f5]")
            chat_log.write("[#c6d0f5]  • all-minilm:latest[/#c6d0f5]")
            chat_log.write("[#c6d0f5]  • embeddinggemma:latest[/#c6d0f5]")
            chat_log.write("[#8bd5ca]💡 Use '/model <name>' to change the current model[/#8bd5ca]")
            input_widget.value = ""
            return

        # Store user message for copying
        self.last_user_message = user_msg

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

        # Create thinking panel
        thinking_steps = []

        def add_thinking_step(step: str):
            thinking_steps.append(step)
            content = "\n".join([f"• {s}" for s in thinking_steps])
            chat_log.write(Panel(
                content,
                title="[bold #ca9ee6]🧠 Thinking...[/bold #ca9ee6]",
                border_style="#ca9ee6"
            ))

        try:
            # Step 1: Embedding query
            add_thinking_step("📝 Embedding your question...")

            # Step 2: Query RAG for relevant context
            add_thinking_step("🔍 Searching document database...")
            rag_start = time.time()

            rag_response = await self.client.post(
                f"{POLARS_API}/query",
                json={"query": user_msg, "top_k": 5}
            )
            rag_response.raise_for_status()
            rag_data = rag_response.json()
            rag_time = int((time.time() - rag_start) * 1000)

            context_chunks = rag_data.get("chunks", [])

            # Build context string
            if context_chunks:
                add_thinking_step(f"✅ Found {len(context_chunks)} relevant chunks ({format_duration(rag_time)})")

                # Show chunk details
                chunk_details = []
                for i, chunk in enumerate(context_chunks, 1):
                    dist = chunk.get('distance', 0)
                    similarity = (1 - dist) * 100  # Convert distance to similarity percentage
                    chunk_details.append(
                        f"{i}. {chunk['title']} (similarity: {similarity:.1f}%)\n"
                        f"   Preview: {chunk['text'][:100]}..."
                    )

                chat_log.write(Panel(
                    "\n\n".join(chunk_details),
                    title=f"[bold #e5c890]📚 Retrieved Context ({len(context_chunks)} chunks)[/bold #e5c890]",
                    border_style="#e5c890"
                ))

                context = "\n\n".join([
                    f"[Document: {chunk['title']}]\n{chunk['text']}"
                    for chunk in context_chunks
                ])
            else:
                add_thinking_step("⚠️ No relevant context found")
                context = "No relevant documents found."

            # Step 3: Send to Ollama
            add_thinking_step(f"💭 Generating response with {CHAT_MODEL}...")

            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use the context to answer the user's question. If the context doesn't contain "
                "relevant information, say so honestly."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_msg}"}
            ]

            # Stream response from Ollama
            response_text = ""
            llm_start = time.time()

            async with self.client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/chat",
                json={"model": CHAT_MODEL, "messages": messages, "stream": True}
            ) as stream:
                # Clear thinking panel
                chat_log.clear()

                # Re-add user message
                chat_log.write(Panel(
                    user_msg,
                    title="[bold cyan]You[/bold cyan]",
                    border_style="cyan"
                ))

                # Show context summary
                if context_chunks:
                    chat_log.write(Panel(
                        f"✓ Using {len(context_chunks)} relevant chunks from database",
                        title="[bold #e5c890]📚 Context[/bold #e5c890]",
                        border_style="#e5c890"
                    ))

                # Stream tokens
                async for line in stream.aiter_lines():
                    if line.strip():
                        import json
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            response_text += chunk["message"]["content"]

            llm_time = int((time.time() - llm_start) * 1000)
            total_time = int((time.time() - self.query_start_time) * 1000)

            # Update status bar
            status_bar.last_query_time = format_duration(total_time)

            # Store for copy
            self.last_response = response_text

            # Show final response with timing
            chat_log.write(Panel(
                Markdown(response_text),
                title=f"[bold #a6d189]🤖 Assistant[/bold #a6d189] (generated in {format_duration(llm_time)}) [dim #838ba7](Ctrl+Y to copy)[/dim #838ba7]",
                border_style="#a6d189",
                subtitle=f"[dim #838ba7]Total: {format_duration(total_time)} | RAG: {format_duration(rag_time)} | LLM: {format_duration(llm_time)}[/dim #838ba7]"
            ))

        except httpx.HTTPError as e:
            chat_log.write(f"[#e78284]❌ HTTP Error: {e}[/#e78284]")
            status_bar.last_query_time = "ERROR"
        except Exception as e:
            chat_log.write(f"[#e78284]❌ Unexpected error: {e}[/#e78284]")
            status_bar.last_query_time = "ERROR"

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
