"""Modified and inspired by:

- ipdb (https://github.com/gotcha/ipdb)
- pdbr (https://github.com/cansarigol/pdbr)
- jdb (https://github.com/jax-ml/jax)

"""

import asyncio
import cmd
import codeop
import functools
import importlib
import os
import pathlib
import sys
import textwrap
import traceback
import types
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import wadler_lindig as wl
from IPython.core.completer import IPCompleter
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ptutils import IPythonPTCompleter
from IPython.terminal.shortcuts import create_ipython_shortcuts
from jax._src.debugger import core as debugger_core
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.cursor_shapes import CursorShape, CursorShapeConfig
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.shortcuts.prompt import PromptSession
from pygments.token import Token
from rich import box
from rich.console import Console, ConsoleRenderable, group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import PathHighlighter


class ModalCursorShapeConfig(CursorShapeConfig):
    """
    Show cursor shape according to the current input mode.
    """

    def get_cursor_shape(self, application: Application[tp.Any]) -> CursorShape:
        if application.editing_mode == EditingMode.VI:
            if application.vi_state.input_mode in {
                InputMode.NAVIGATION,
            }:
                return CursorShape.BLOCK
            if application.vi_state.input_mode in {
                InputMode.INSERT,
                InputMode.INSERT_MULTIPLE,
            }:
                return CursorShape.BLINKING_BLOCK
            if application.vi_state.input_mode in {
                InputMode.REPLACE,
                InputMode.REPLACE_SINGLE,
            }:
                return CursorShape.UNDERLINE
        elif application.editing_mode == EditingMode.EMACS:
            return CursorShape.BLOCK

        # Default
        return CursorShape.BLINKING_BLOCK


class CliDebugger(cmd.Cmd):
    prompt = "(jdb++) "

    def __init__(
        self,
        frames: list[debugger_core.DebuggerFrame],
        thread_id,
        stdin: tp.IO[str] | None = None,
        stdout: tp.IO[str] | None = None,
        completekey: str = "tab",
    ):
        super().__init__(stdin=stdin, stdout=stdout, completekey=completekey)
        self.use_rawinput = stdin is None
        self.frames = frames
        self.frame_index = 0
        self.thread_id = thread_id
        self.thread_executor = ThreadPoolExecutor(1)

        compl = IPCompleter(shell=self.shell, namespace={}, global_namespace={}, parent=self.shell)
        # add a completer for all the do_ methods
        methods_names = [m[3:] for m in dir(self) if m.startswith("do_")]

        def gen_comp(_, text):
            return [m for m in methods_names if m.startswith(text)]

        newcomp = types.MethodType(gen_comp, compl)
        compl.custom_matchers.insert(0, newcomp)
        # end add completer.

        self.pt_comp = IPythonPTCompleter(compl)

        # setup history only when we start pdb
        if self.shell.debugger_history is None:
            if self.shell.debugger_history_file is not None:
                p = pathlib.Path(self.shell.debugger_history_file).expanduser()
                if not p.exists():
                    p.touch()
                self.debugger_history = FileHistory(os.path.expanduser(str(p)))
            else:
                self.debugger_history = InMemoryHistory()
        else:
            self.debugger_history = self.shell.debugger_history

        self.pt_loop = asyncio.new_event_loop()
        self.pt_app = PromptSession(
            message=(lambda: PygmentsTokens([(Token.Prompt, self.prompt)])),
            editing_mode=getattr(EditingMode, self.shell.editing_mode.upper()),
            key_bindings=create_ipython_shortcuts(self.shell),
            history=self.debugger_history,
            completer=self.pt_comp,
            enable_history_search=True,
            mouse_support=self.shell.mouse_support,
            complete_style=self.shell.pt_complete_style,
            style=getattr(self.shell, "style", None),
            color_depth=self.shell.color_depth,
            vi_mode=True,
            cursor=ModalCursorShapeConfig(),
        )

    @functools.cached_property
    def console(self) -> Console:
        return Console(file=self.stdout, force_terminal=True, force_interactive=True)

    @functools.cached_property
    def shell(self) -> TerminalInteractiveShell:
        save_main = sys.modules["__main__"]
        shell = TerminalInteractiveShell.instance()
        shell.editing_mode = "vi"
        sys.modules["__main__"] = save_main
        return shell

    def _prompt(self):
        """
        In case other prompt_toolkit apps have to run in parallel to this one (e.g. in madbg),
        create_app_session must be used to prevent mixing up between them. According to the prompt_toolkit docs:

        > If you need multiple applications running at the same time, you have to create a separate
        > `AppSession` using a `with create_app_session():` block.
        """
        with create_app_session():
            return self.pt_app.prompt()

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        override the same methods from cmd.Cmd to provide prompt toolkit replacement.
        """
        if not self.use_rawinput:
            raise ValueError("Sorry jdbpp does not support use_rawinput=False")

        # In order to make sure that prompt, which uses asyncio doesn't
        # interfere with applications in which it's used, we always run the
        # prompt itself in a different thread (we can't start an event loop
        # within an event loop). This new thread won't have any event loop
        # running, and here we run our prompt-loop.
        self.preloop()

        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                print(self.intro, file=self.stdout)
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    self.pt_comp.ipy_completer.namespace = self.current_frame.locals
                    self.pt_comp.ipy_completer.global_namespace = self.current_frame.globals

                    try:
                        line = self.thread_executor.submit(self._prompt).result()
                    except EOFError:
                        line = "EOF"

                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        except Exception:
            raise

    def precmd(self, line):
        """Perform useful escapes on the command before it is executed."""

        if line.endswith("??"):
            line = "pinfo2 " + line[:-2]
        elif line.endswith("?"):
            line = "pinfo " + line[:-1]

        line = super().precmd(line)

        return line

    def onecmd(self, line: str) -> bool:
        """
        Invokes 'run_magic()' if the line starts with a '%'.
        The loop stops of this function returns True.
        (unless an overridden 'postcmd()' behaves differently)
        """
        try:
            line = line.strip()
            if line.startswith("%"):
                if line.startswith("%%"):
                    self.console.print(
                        "[bold red]Cell magics (multiline) are not yet supported. Use a single '%' instead.[/bold red]",
                    )
                    return False
                self.run_magic(line[1:])
                return False
            return super().onecmd(line)

        except:  # noqa: E722
            self.error_message()
            return False

    @property
    def current_frame(self):
        return self.frames[self.frame_index]

    def evaluate(self, expr):
        curr_frame = self.frames[self.frame_index]
        return eval(expr, curr_frame.globals, curr_frame.locals)

    def default(self, line):
        # Modified from pdb
        if line[:1] == "!":
            line = line[1:].strip()
        currframe = self.frames[self.frame_index]
        locals = currframe.locals
        globals = currframe.globals
        try:
            buffer = line
            if (code := codeop.compile_command(line + "\n", "<stdin>", "single")) is None:
                # Multi-line mode
                with self._disable_command_completion():
                    buffer = line
                    continue_prompt = "...   "
                    while (code := codeop.compile_command(buffer, "<stdin>", "single")) is None:
                        if self.use_rawinput:
                            try:
                                line = input(continue_prompt)
                            except (EOFError, KeyboardInterrupt):
                                self.lastcmd = ""
                                print("\n")
                                return
                        else:
                            self.stdout.write(continue_prompt)
                            self.stdout.flush()
                            line = self.stdin.readline()
                            if not len(line):
                                self.lastcmd = ""
                                self.stdout.write("\n")
                                self.stdout.flush()
                                return
                            else:
                                line = line.rstrip("\r\n")
                        buffer += "\n" + line
                    self.lastcmd = buffer
            save_stdout = sys.stdout
            save_stdin = sys.stdin
            save_displayhook = sys.displayhook
            try:
                sys.stdin = self.stdin
                sys.stdout = self.stdout
                sys.displayhook = self.displayhook
                if not self.exec_in_closure(buffer, globals, locals):
                    exec(code, globals, locals)
            finally:
                sys.stdout = save_stdout
                sys.stdin = save_stdin
                sys.displayhook = save_displayhook
        except:  # noqa: E722
            self.error_message()

    def do_v(self, _):
        # Modified from pdbr
        try:
            variables = [
                (k, wl.pformat(v, short_arrays=True), f"{type(v).__module__}.{type(v).__name__}")
                for k, v in self.current_frame.locals.items()
                if not k.startswith("__") and k != "jdbpp"
            ]
        except AttributeError:
            return

        table = Table(title="List of local variables", box=box.MINIMAL)

        table.add_column("Variable", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Type", style="green")
        [table.add_row(variable, value, _type) for variable, value, _type in variables]
        self.console.print(table)

    def exec_in_closure(self, source, globals, locals):
        """Run source code in closure so code object created within source
        can find variables in locals correctly

        returns True if the source is executed, False otherwise
        """

        # Determine if the source should be executed in closure. Only when the
        # source compiled to multiple code objects, we should use this feature.
        # Otherwise, we can just raise an exception and normal exec will be used.

        code = compile(source, "<string>", "exec")
        if not any(isinstance(const, types.CodeType) for const in code.co_consts):
            return False

        # locals could be a proxy which does not support pop
        # copy it first to avoid modifying the original locals
        locals_copy = dict(locals)

        locals_copy["__jdb_eval__"] = {"result": None, "write_back": {}}

        # If the source is an expression, we need to print its value
        try:
            compile(source, "<string>", "eval")
        except SyntaxError:
            pass
        else:
            source = "__jdb_eval__['result'] = " + source

        # Add write-back to update the locals
        source = (
            "try:\n" + textwrap.indent(source, "  ") + "\n" + "finally:\n" + "  __jdb_eval__['write_back'] = locals()"
        )

        # Build a closure source code with freevars from locals like:
        # def __jdb_outer():
        #   var = None
        #   def __jdb_scope():  # This is the code object we want to execute
        #     nonlocal var
        #     <source>
        #   return __jdb_scope.__code__
        source_with_closure = (
            "def __jdb_outer():\n"
            + "\n".join(f"  {var} = None" for var in locals_copy)
            + "\n"
            + "  def __jdb_scope():\n"
            + "\n".join(f"    nonlocal {var}" for var in locals_copy)
            + "\n"
            + textwrap.indent(source, "    ")
            + "\n"
            + "  return __jdb_scope.__code__"
        )

        # Get the code object of __jdb_scope()
        # The exec fills locals_copy with the __jdb_outer() function and we can call
        # that to get the code object of __jdb_scope()
        ns = {}
        try:
            exec(source_with_closure, {}, ns)
        except Exception:
            return False
        code = ns["__jdb_outer"]()

        cells = tuple(types.CellType(locals_copy.get(var)) for var in code.co_freevars)

        try:
            exec(code, globals, locals_copy, closure=cells)
        except Exception:
            return False

        # get the data we need from the statement
        jdb_eval = locals_copy["__jdb_eval__"]

        # __jdb_eval__ should not be updated back to locals
        jdb_eval["write_back"].pop("__jdb_eval__")

        # Write all local variables back to locals
        locals.update(jdb_eval["write_back"])
        eval_result = jdb_eval["result"]
        if eval_result is not None:
            self.console.print(eval_result)

        return True

    def displayhook(self, obj):
        """Custom displayhook for the exec in default(), which prevents
        assignment of the _ variable in the builtins.
        """
        # reproduce the behavior of the standard displayhook, not printing None
        if obj is not None:
            self.console.print(obj)

    @contextmanager
    def _disable_command_completion(self):
        completenames = self.completenames
        try:
            self.completenames = self.completedefault  # type: ignore
            yield
        finally:
            self.completenames = completenames
        return

    @functools.cache
    @group()
    def render_stack(self, frame_index: int) -> tp.Generator[ConsoleRenderable, None, None]:
        path_highlighter = PathHighlighter()

        for frame in reversed(self.frames[frame_index:]):
            yield Text.assemble(
                Text("  File ", style="pygments.string"),
                path_highlighter(Text(frame.filename, style="pygments.string")),
                Text(f"({frame.lineno})", style="pygments.string"),
            )
            if frame.offset is None:
                yield Text("    <no source>", style="pygments.text")
            else:
                num_lines = 2
                yield self.render_source("\n".join(frame.source), frame.offset, frame.lineno, num_lines)

    def print_backtrace(self):
        self.console.print(
            Panel(
                self.render_stack(self.frame_index),
                title="[traceback.title]Traceback [dim](most recent call last)",
                border_style="traceback.border",
                expand=True,
                padding=(0, 1),
            )
        )

    @functools.cache
    @group()
    def render_source(
        self, source: str, offset: int | None, lineno: int, num_lines: int
    ) -> tp.Generator[ConsoleRenderable, None, None]:
        if offset is None:
            return

        yield Syntax(
            source,
            lexer="python",
            theme="ansi_light",
            line_numbers=True,
            start_line=lineno - offset,
            line_range=(offset - num_lines, offset + num_lines),
            highlight_lines={lineno},
            padding=(0, 4),
        )

    def print_context(self, num_lines: int | None = 2):
        curr_frame = self.frames[self.frame_index]
        if num_lines is None:
            num_lines = len(curr_frame.source)

        path_highlighter = PathHighlighter()

        self.console.print(
            Text.assemble(
                Text("> ", style="pygments.string"),
                path_highlighter(Text(curr_frame.filename, style="pygments.string")),
                Text(f"({curr_frame.lineno})", style="pygments.string"),
            )
        )

        if curr_frame.source:
            assert curr_frame.offset is not None
            self.console.print(
                self.render_source("\n".join(curr_frame.source), curr_frame.offset, curr_frame.lineno, num_lines)
            )
            self.shell.hooks.synchronize_with_editor(curr_frame.filename, curr_frame.lineno, 0)

    def error_message(self):
        exc_info = sys.exc_info()[:2]
        msg = traceback.format_exception_only(*exc_info)[-1].strip()
        self.console.print(f"*** [bold red]{msg}[/bold red]")

    def do_p(self, arg):
        try:
            self.console.print(repr(self.evaluate(arg)))
        except:  # noqa: E722
            self.error_message()

    def do_pp(self, arg):
        try:
            self.console.print(Pretty(self.evaluate(arg)))
        except:  # noqa: E722
            self.error_message()

    def do_up(self, _):
        """u(p). Move up a stack frame."""
        if self.frame_index == len(self.frames) - 1:
            self.console.print("At topmost frame.")
        else:
            self.frame_index += 1
        self.print_context()

    do_u = do_up

    def do_down(self, _):
        """d(own). Move down a stack frame."""
        if self.frame_index == 0:
            self.console.print("At bottommost frame.")
        else:
            self.frame_index -= 1
        self.print_context()

    do_d = do_down

    def do_list(self, arg):
        self.print_context(num_lines=int(arg) if arg else 5)

    do_l = do_list

    def do_longlist(self, _):
        self.print_context(num_lines=50)

    do_ll = do_longlist

    def do_continue(self, _):
        return True

    do_c = do_cont = do_continue

    def do_quit(self, _):
        os._exit(0)

    do_q = do_EOF = do_exit = do_quit

    def do_where(self, _):
        self.print_backtrace()

    do_w = do_bt = do_where

    def do_interact(self, _):
        cfg = self.shell.config.copy()
        cfg.TerminalInteractiveShell.confirm_exit = False  # type: ignore

        embed = InteractiveShellEmbed(config=cfg, parent=self.shell, banner1="", exitmsg="")

        ns = self.current_frame.globals.copy()
        ns.update(self.current_frame.locals)
        embed(local_ns=ns)

    do_i = do_interact

    def do_s(self, arg):
        code = f"jax.eval_shape(lambda: {arg})"
        locals = self.current_frame.locals
        globals = self.current_frame.globals

        save_stdout = sys.stdout
        save_stdin = sys.stdin
        save_displayhook = sys.displayhook
        try:
            sys.stdin = self.stdin
            sys.stdout = self.stdout
            sys.displayhook = self.displayhook
            if not self.exec_in_closure(code, globals | {"jax": importlib.import_module("jax")}, locals):
                exec(code, globals, locals)
        except:  # noqa: E722
            self.error_message()
        finally:
            sys.stdout = save_stdout
            sys.stdin = save_stdin
            sys.displayhook = save_displayhook

    do_shape = do_s

    def do_pdef(self, arg):
        """Print the call signature for any callable object.

        The debugger interface to %pdef"""
        namespaces = [
            ("Locals", self.current_frame.locals),
            ("Globals", self.current_frame.globals),
        ]
        self.shell.find_line_magic("pdef")(arg, namespaces=namespaces)  # type: ignore

    def do_pdoc(self, arg):
        """Print the docstring for an object.

        The debugger interface to %pdoc."""
        namespaces = [
            ("Locals", self.current_frame.locals),
            ("Globals", self.current_frame.globals),
        ]
        self.shell.find_line_magic("pdoc")(arg, namespaces=namespaces)  # type: ignore

    def do_pfile(self, arg):
        """Print (or run through pager) the file where an object is defined.

        The debugger interface to %pfile.
        """
        namespaces = [
            ("Locals", self.current_frame.locals),
            ("Globals", self.current_frame.globals),
        ]
        self.shell.find_line_magic("pfile")(arg, namespaces=namespaces)  # type: ignore

    def do_pinfo(self, arg):
        """Provide detailed information about an object.

        The debugger interface to %pinfo, i.e., obj?."""
        namespaces = [
            ("Locals", self.current_frame.locals),
            ("Globals", self.current_frame.globals),
        ]
        self.shell.find_line_magic("pinfo")(arg, namespaces=namespaces)  # type: ignore

    def do_pinfo2(self, arg):
        """Provide extra detailed information about an object.

        The debugger interface to %pinfo2, i.e., obj??."""
        namespaces = [
            ("Locals", self.current_frame.locals),
            ("Globals", self.current_frame.globals),
        ]
        self.shell.find_line_magic("pinfo2")(arg, namespaces=namespaces)  # type: ignore

    def do_psource(self, arg):
        """Print (or run through pager) the source code for an object."""
        namespaces = [
            ("Locals", self.current_frame.locals),
            ("Globals", self.current_frame.globals),
        ]
        self.shell.find_line_magic("psource")(arg, namespaces=namespaces)  # type: ignore

    def run_magic(self, line) -> str:
        """
        Parses the line and runs the appropriate magic function.
        Assumes that the line is without a leading '%'.
        """
        magic_name, arg, line = self.parseline(line)
        if hasattr(self, f"do_{magic_name}"):
            # We want to use do_{magic_name} methods if defined.
            # This is indeed the case with do_pdef, do_pdoc etc,
            # which are defined by our base class (IPython.core.debugger.Pdb).
            result = getattr(self, f"do_{magic_name}")(arg)
        else:
            magic_fn = self.shell.find_line_magic(magic_name)
            if not magic_fn:
                self.console.print(f"[bold red]Line Magic %{magic_name} not found[/bold red]")
                return ""
            if magic_name in ("time", "timeit"):
                result = magic_fn(
                    arg,
                    local_ns={**self.current_frame.locals, **self.current_frame.globals},
                )
            else:
                result = magic_fn(arg)
        if result:
            result = str(result)
            self.console.print(result)
        return ""

    def run(self):
        while True:
            try:
                self.cmdloop()
                break
            except KeyboardInterrupt:
                self.console.print("--KeyboardInterrupt--")


def run_debugger(frames: list[debugger_core.DebuggerFrame], thread_id: int | None, **kwargs: tp.Any):
    CliDebugger(frames, thread_id, **kwargs).run()


debugger_core.register_debugger("clipp", run_debugger, 1)


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    @jax.jit
    def f(x):
        a = jnp.arange(10)
        a *= x

        jax.debug.breakpoint()

        return a

    f(1)
