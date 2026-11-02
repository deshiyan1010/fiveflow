"""Replace only complete Fiveflow blocks while preserving shell configuration."""
import os
from pathlib import Path
import stat
import shlex
import sys
import tempfile

START = '# >>> fiveflow >>>'
END = '# <<< fiveflow <<<'


def replace_block(content, block=''):
    result = []
    inside = False
    for line in content.splitlines(keepends=True):
        marker = line.strip()
        if marker == START:
            if inside:
                raise ValueError('Nested Fiveflow block; shell configuration left unchanged')
            inside = True
        elif marker == END:
            if not inside:
                raise ValueError('Unmatched Fiveflow block; shell configuration left unchanged')
            inside = False
        elif not inside:
            result.append(line)
    if inside:
        raise ValueError('Incomplete Fiveflow block; shell configuration left unchanged')
    remaining = ''.join(result)
    if block:
        remaining = remaining.rstrip('\n') + '\n\n' + block.strip('\n') + '\n'
    return remaining


def write_config(path, block=''):
    path = Path(path).expanduser().resolve()  # Keep symlinked dotfiles intact.
    content = path.read_text() if path.exists() else ''
    updated = replace_block(content, block)
    metadata = path.stat() if path.exists() else None
    mode = stat.S_IMODE(metadata.st_mode) if metadata else 0o600
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as temporary:
            temp_name = temporary.name
            temporary.write(updated)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temp_name, mode)
        if metadata is not None and os.geteuid() == 0:
            os.chown(temp_name, metadata.st_uid, metadata.st_gid)
        if path.exists() and path.read_text() != content:
            raise ValueError('Shell configuration changed while updating; retry')
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def command_block(install_dir, shell_rc, whisper_cache, gemma_cache):
    root = Path(install_dir)
    python = shlex.quote(str(root / 'stt_env/bin/python'))
    entry = shlex.quote(str(root / 'stt_pipeline.py'))
    update = shlex.quote(str(root / 'fiveflow/updates.py'))
    config = shlex.quote(str(root / 'fiveflow/shell_config.py'))
    rc = shlex.quote(str(shell_rc))
    return f"""{START}
fiveflow() {{
    {python} {entry} \"$@\"
}}
fiveflow-update() {{
    {python} {update}
}}
fiveflow-remove() {{
    echo '[fiveflow] Uninstalling...'
    if [ -w {rc} ]; then
        {python} {config} --remove {rc} || return 1
    else
        sudo {python} {config} --remove {rc} || return 1
    fi
    rm -rf {shlex.quote(str(root))} || return 1
    rm -rf {shlex.quote(str(whisper_cache))} || return 1
    rm -rf {shlex.quote(str(gemma_cache))} || return 1
    unset -f fiveflow fiveflow-update fiveflow-remove
    echo '[fiveflow] Fully removed.'
}}
{END}
"""

def main():
    try:
        if sys.argv[1] == '--remove':
            write_config(sys.argv[2])
        elif sys.argv[1] == '--install':
            write_config(sys.argv[2], command_block(sys.argv[3], sys.argv[2], sys.argv[4], sys.argv[5]))
        else:
            write_config(sys.argv[1], Path(sys.argv[2]).read_text())
    except (OSError, ValueError, IndexError) as exc:
        print(f'[fiveflow] Could not update shell configuration: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
